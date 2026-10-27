//! Safeguarded squared extrapolation of a smooth fixed-point map (SQUAREM).
//!
//! A cycle evaluates x1=F(x0), x2=F(x1), r=x1-x0, v=x2-2*x1+x0,
//! alpha=-sqrt((r'r)/(v'v)), and candidate=x0-2*alpha*r+alpha^2*v.
//! The accelerated candidate is F(candidate), subject to positivity and
//! exact likelihood ascent relative to the *two-step* unaccelerated path.
//! If extrapolation is unsuitable, we accept x2. This is a conservative
//! SQUAREM variant, not a promise of faster convergence for every dataset.
//!
//! All calls to F and to the objective are counted, including rejected work.
//! max_iter counts outer SQUAREM cycles, not calls to F. No R API calls occur
//! in this module. The likelihood acceptance slack matches the base engines.

use extendr_api::prelude::*;

fn fail(message: impl Into<String>) -> Error { Error::Other(message.into()) }

pub struct ResultFit<S> {
    pub theta: Vec<f64>,
    pub state: S,
    pub history: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub reason: &'static str,
    pub relative_change: f64,
    pub relative_ll_change: f64,
    pub fp_evals: usize,
    pub likelihood_evals: usize,
    pub attempts: usize,
    pub accepted: usize,
    pub rejected: usize,
}

fn relative_delta(old: &[f64], new: &[f64]) -> f64 {
    old.iter().zip(new.iter()).map(|(a,b)| (a-b).abs() / (1.0+a.abs()))
        .fold(0.0_f64, f64::max)
}

// One base F step, with the same monotonicity/feasibility safeguard as 5C.
// Both F and the objective are counted, including rejected feasibility tests.
fn base_step<S, E, F, L>(
    theta: &[f64], state: &S, eval: &mut E, propose: &mut F, loglik: &L,
    fp_evals: &mut usize, likelihood_evals: &mut usize,
) -> extendr_api::Result<(Vec<f64>, S)>
where E: FnMut(&[f64]) -> extendr_api::Result<S>,
      F: FnMut(&[f64], &S) -> extendr_api::Result<Vec<f64>>,
      L: Fn(&S) -> f64,
{
    *fp_evals += 1;
    let mut candidate = propose(theta, state)?;
    if candidate.len() != theta.len() { return Err(fail("VCM fixed-point map changed dimension.")); }
    let slack = 1e-9 * (1.0 + loglik(state).abs());
    for _ in 0..=25 {
        *likelihood_evals += 1;
        if let Ok(next) = eval(&candidate) {
            if loglik(&next).is_finite() && loglik(&next) + slack >= loglik(state) {
                return Ok((candidate, next));
            }
        }
        for (a,b) in theta.iter().zip(candidate.iter_mut()) {
            *b = *a + 0.5 * (*b - *a);
        }
    }
    Err(fail("VCM fixed-point step was infeasible or non-monotone after backtracking."))
}

#[allow(clippy::too_many_arguments)]
pub fn run<S, E, F, L, K>(
    mut theta: Vec<f64>, mut state: S, max_iter: usize,
    tol: f64, stop_rule: &str, kkt_tol: f64,
    mut eval: E, mut propose: F, loglik: L, kkt: K,
) -> extendr_api::Result<ResultFit<S>>
where E: FnMut(&[f64]) -> extendr_api::Result<S>,
      F: FnMut(&[f64], &S) -> extendr_api::Result<Vec<f64>>,
      L: Fn(&S) -> f64,
      K: Fn(&S, &[f64]) -> f64,
{
    let mut likelihood_evals = 1usize; // initial state was evaluated by caller
    let mut fp_evals = 0usize;
    let mut attempts = 0usize;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut iterations = 0usize;
    let mut relative_change = f64::INFINITY;
    let mut relative_ll_change = f64::INFINITY;
    let mut history = vec![loglik(&state)];
    let mut reason = "max_iter";
    let converged;
    loop {
        let scaled_kkt = kkt(&state, &theta);
        if !scaled_kkt.is_finite() { return Err(fail("Nonfinite SQUAREM KKT residual.")); }
        let param_ok = iterations > 0 && relative_change < tol;
        let kkt_ok = scaled_kkt <= kkt_tol;
        let satisfied = match stop_rule {
            "parameter" => param_ok, "kkt" => kkt_ok, "both" => param_ok && kkt_ok,
            _ => return Err(fail("Invalid SQUAREM stopping rule.")),
        };
        if satisfied {
            converged = true;
            reason = match stop_rule { "parameter" => "parameter", "kkt" => "kkt", _ => "both" };
            break;
        }
        if iterations >= max_iter { converged = false; break; }

        // Compute the unaccelerated reference F(F(theta)).
        let (first_theta, first_state) = base_step(
            &theta, &state, &mut eval, &mut propose, &loglik,
            &mut fp_evals, &mut likelihood_evals,
        )?;
        let (second_theta, second_state) = base_step(
            &first_theta, &first_state, &mut eval, &mut propose, &loglik,
            &mut fp_evals, &mut likelihood_evals,
        )?;
        let r: Vec<f64> = first_theta.iter().zip(theta.iter())
            .map(|(b,a)| b-a).collect();
        let v: Vec<f64> = second_theta.iter().zip(first_theta.iter()).zip(theta.iter())
            .map(|((c,b),a)| c-2.0*b+a).collect();
        let rr: f64 = r.iter().map(|x| x*x).sum();
        let vv: f64 = v.iter().map(|x| x*x).sum();
        let mut new_theta = second_theta;
        let mut new_state = second_state;

        // The classical negative SQUAREM step length has |alpha|>=1.
        // Clip |alpha| to 8 to prevent very large speculative steps.
        if rr.is_finite() && vv.is_finite() && rr > 0.0 && vv > 1e-30 * (1.0+rr) {
            let initial_scale = (rr/vv).sqrt().clamp(1.0, 8.0);
            // Backtrack the *extrapolation magnitude* if an aggressive
            // speculative step is infeasible or fails the likelihood test.
            // A rejected SQUAREM cycle still accepts ordinary F(F(theta)).
            if initial_scale > 1.001 {
                attempts += 1;
                let mut scale = initial_scale;
                let mut accel: Option<(Vec<f64>, S)> = None;
                for _ in 0..6 {
                    if scale <= 1.001 { break; }
                    let extrap: Vec<f64> = theta.iter().enumerate()
                        .map(|(k,a)| a + 2.0*scale*r[k] + scale*scale*v[k])
                        .collect();
                    let feasible = extrap.iter().all(|x| x.is_finite() && *x >= 0.0);
                    if feasible {
                        likelihood_evals += 1;
                        if let Ok(extrap_state) = eval(&extrap) {
                            fp_evals += 1;
                            if let Ok(proposal) = propose(&extrap, &extrap_state) {
                                if proposal.len() == theta.len() &&
                                   proposal.iter().all(|x| x.is_finite() && *x >= 0.0) {
                                    likelihood_evals += 1;
                                    if let Ok(candidate_state) = eval(&proposal) {
                                        let ll_base = loglik(&new_state);
                                        let slack = 1e-9 * (1.0 + ll_base.abs());
                                        if loglik(&candidate_state).is_finite() &&
                                           loglik(&candidate_state) + slack >= ll_base {
                                            accel = Some((proposal, candidate_state));
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    scale = 1.0 + 0.5*(scale-1.0);
                }
                if let Some((a,b)) = accel {
                    new_theta = a;
                    new_state = b;
                    accepted += 1;
                } else { rejected += 1; }
            }
        }

        relative_change = relative_delta(&theta, &new_theta);
        relative_ll_change = (loglik(&new_state)-loglik(&state)).abs()
            / (1.0+loglik(&state).abs());
        theta = new_theta;
        state = new_state;
        history.push(loglik(&state));
        iterations += 1;
    }
    Ok(ResultFit {
        theta, state, history, iterations, converged, reason,
        relative_change, relative_ll_change, fp_evals, likelihood_evals,
        attempts, accepted, rejected,
    })
}
