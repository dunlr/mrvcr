//! Two-kernel Gaussian VCM spectral backend (Zhou, Hu, Zhou, Lange 2019, Section 3.1).
//!
//! The R frontend performs a one-time generalized eigenvalue decomposition
//! U' K1 U = diag(d), U' K2 U = I, for a positive-definite K2.
//! Here we fit the EXACT ML/REML objective in transformed coordinates:
//! v_i(theta) = theta[0] * d_i + theta[1].
//! No dense n-by-n covariance or inverse is formed in this iteration loop.
//! The same EM/MM mappings, KKT scaling, convergence, and backtracking rules
//! as gaussian_em.rs are used. The fixed-effects coefficients are unchanged by
//! the invertible transformation. Original-scale fitted/residual vectors are
//! reconstructed by the R frontend.

use extendr_api::prelude::*;
use faer::{Mat, Side};
use faer::prelude::Solve;
use crate::core::{storage::dense_matrix_from_robj, Pet, ApproximationMethod};
use crate::core::method::NumericalOptimizer;

fn fail(message: impl Into<String>) -> Error { Error::Other(message.into()) }

fn logdet_cholesky(l: faer::MatRef<'_, f64>) -> extendr_api::Result<f64> {
    let mut sum = 0.0;
    for i in 0..l.nrows() {
        let v = l[(i, i)];
        if !v.is_finite() || v <= 0.0 { return Err(fail("Invalid spectral GLS Cholesky pivot.")); }
        sum += 2.0 * v.ln();
    }
    Ok(sum)
}

struct State {
    loglik: f64,
    beta: Vec<f64>,
    score: [f64; 2],
    quadratic_trace: [(f64, f64); 2],
}

// All arithmetic, including REML and scores, is in transformed coordinates.
// Original covariance logdet = logdet(K2) + sum_i log(theta1*d_i+theta2).
fn evaluate(y: &[f64], x: &Mat<f64>, d: &[f64], theta: &[f64],
            logdet_k2: f64, restricted: bool) -> extendr_api::Result<State> {
    let n = y.len();
    let p = x.ncols();
    let mut weights = vec![0.0; n];
    let mut logdet_v = logdet_k2;
    for i in 0..n {
        let v = theta[0].mul_add(d[i], theta[1]);
        if !v.is_finite() || v <= 0.0 {
            return Err(fail("Spectral covariance has nonpositive/nonfinite eigenvalue."));
        }
        weights[i] = 1.0 / v;
        logdet_v += v.ln();
    }
    let gram = Mat::<f64>::from_fn(p, p, |j, k| {
        (0..n).map(|i| x[(i, j)] * weights[i] * x[(i, k)]).sum::<f64>()
    });
    let rhs = Mat::<f64>::from_fn(p, 1, |j, _| {
        (0..n).map(|i| x[(i, j)] * weights[i] * y[i]).sum::<f64>()
    });
    let chol = gram.llt(Side::Lower).map_err(|_| fail(
        "Spectral X' W X is singular; the fixed-effects design must have full rank."
    ))?;
    let logdet_x = if restricted { logdet_cholesky(chol.L())? } else { 0.0 };
    let beta_mat = chol.solve(&rhs);
    let beta: Vec<f64> = (0..p).map(|j| beta_mat[(j, 0)]).collect();
    let mut alpha = vec![0.0; n];
    let mut quadratic = 0.0;
    for i in 0..n {
        let fitted: f64 = (0..p).map(|j| x[(i, j)] * beta[j]).sum();
        let residual = y[i] - fitted;
        alpha[i] = weights[i] * residual;
        quadratic += residual * alpha[i];
    }
    let dof = if restricted { n - p } else { n };
    let loglik = -0.5 * ((dof as f64) * (2.0 * std::f64::consts::PI).ln()
                            + logdet_v + logdet_x + quadratic);
    if !loglik.is_finite() { return Err(fail("Nonfinite spectral Gaussian likelihood.")); }

    // Diagonal of the transformed REML projector, without forming n*n P.
    // P' = W - W X (X' W X)^-1 X' W.
    let inverse_gram = if restricted { Some(chol.solve(&Mat::<f64>::identity(p, p))) }
                       else { None };
    let mut q = [0.0; 2];
    let mut tr = [0.0; 2];
    for i in 0..n {
        let proj = if let Some(ref inv_gram) = inverse_gram {
            let mut leverage = 0.0;
            for j in 0..p {
                for k in 0..p {
                    leverage += x[(i, j)] * inv_gram[(j, k)] * x[(i, k)];
                }
            }
            weights[i] - weights[i] * weights[i] * leverage
        } else { weights[i] };
        q[0] += d[i] * alpha[i] * alpha[i];
        q[1] += alpha[i] * alpha[i];
        tr[0] += d[i] * proj;
        tr[1] += proj;
    }
    let score = [0.5 * (q[0] - tr[0]), 0.5 * (q[1] - tr[1])];
    Ok(State {
        loglik, beta, score,
        quadratic_trace: [(q[0], tr[0]), (q[1], tr[1])],
    })
}

struct Diagnostics {
    raw: f64,
    scaled: f64,
    on_boundary: [bool; 2],
}

fn diagnostics(state: &State, theta: &[f64], boundary_tol: f64) -> Diagnostics {
    let max_theta = theta[0].max(theta[1]);
    let threshold = boundary_tol * (1.0 + max_theta);
    let mut raw: f64 = 0.0;
    let mut scaled: f64 = 0.0;
    let mut on_boundary = [false; 2];
    for k in 0..2 {
        let boundary = theta[k] <= threshold;
        on_boundary[k] = boundary;
        let violation = if boundary { state.score[k].max(0.0) } else { state.score[k].abs() };
        let (q, tr) = state.quadratic_trace[k];
        raw = raw.max(violation);
        scaled = scaled.max(violation / (1.0 + 0.5 * (q.abs() + tr.abs())));
    }
    Diagnostics { raw, scaled, on_boundary }
}

#[allow(clippy::too_many_arguments)]
pub fn ffi_fit_spectral(y_obj: Robj, x_obj: Robj, eig_obj: Robj,
                        ranks_obj: Robj, theta_obj: Robj, logdet_reference: f64,
                        pet_str: &str, method_str: &str, max_iter: i32, tol: f64,
                        stop_rule_str: &str, kkt_tol: f64, boundary_tol: f64)
                        -> extendr_api::Result<List> {
    let pet = Pet::try_from(pet_str)?;
    let restricted = match pet {
        Pet::Reml => true,
        Pet::Mle => false,
        _ => return Err(fail("Spectral Gaussian VCM supports PET = MLE or REML.")),
    };
    let method = ApproximationMethod::try_from(method_str)?;
    let use_mm = match method {
        ApproximationMethod::Optimization(NumericalOptimizer::MmAlgorithm) => true,
        ApproximationMethod::Optimization(NumericalOptimizer::EmAlgorithm) => false,
        _ => return Err(fail("Spectral Gaussian VCM supports method = EM or MM.")),
    };
    let stop_rule = match stop_rule_str {
        "parameter" => "parameter", "kkt" => "kkt", "both" => "both",
        _ => return Err(fail("stop_rule must be parameter, kkt or both.")),
    };
    if max_iter <= 0 || !tol.is_finite() || tol <= 0.0 ||
       !kkt_tol.is_finite() || kkt_tol <= 0.0 ||
       !boundary_tol.is_finite() || boundary_tol < 0.0 ||
       !logdet_reference.is_finite() {
        return Err(fail("Invalid spectral VCM tolerances, maximum iterations, or logdet reference."));
    }
    let x = dense_matrix_from_robj(&x_obj, "transformed X")?;
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 || (restricted && p >= n) {
        return Err(fail("Spectral VCM requires n > p >= 1 for REML."));
    }
    let y = y_obj.as_real_slice().ok_or_else(|| fail("Transformed response must be a double vector."))?;
    let d = eig_obj.as_real_slice().ok_or_else(|| fail("Generalized eigenvalues must be a double vector."))?;
    let theta0 = theta_obj.as_real_slice().ok_or_else(|| fail("theta_init must be a double vector."))?;
    let ranks = ranks_obj.as_integer_slice().ok_or_else(|| fail("Kernel ranks must be integer."))?;
    if y.len() != n || d.len() != n || theta0.len() != 2 || ranks.len() != 2 ||
       ranks.iter().any(|r| *r <= 0 || (*r as usize) > n) ||
       theta0.iter().any(|t| !t.is_finite() || *t <= 0.0) ||
       d.iter().any(|v| !v.is_finite() || *v < 0.0) ||
       y.iter().any(|v| !v.is_finite()) {
        return Err(fail("Invalid spectral transformed data, kernel ranks or starting values."));
    }
    let mut theta = theta0.to_vec();
    let mut state = evaluate(y, &x, d, &theta, logdet_reference, restricted)?;
    let mut history = vec![state.loglik];
    let mut iterations = 0usize;
    let mut converged;
    let mut reason = "max_iter";
    let mut relative_change = f64::INFINITY;
    let mut relative_ll_change = f64::INFINITY;
    let mut final_diag;
    loop {
        final_diag = diagnostics(&state, &theta, boundary_tol);
        if !final_diag.scaled.is_finite() { return Err(fail("Nonfinite spectral KKT violation.")); }
        let param_ok = iterations > 0 && relative_change < tol;
        let kkt_ok = final_diag.scaled <= kkt_tol;
        converged = match stop_rule {
            "parameter" => param_ok, "kkt" => kkt_ok, _ => param_ok && kkt_ok,
        };
        if converged {
            reason = stop_rule;
            break;
        }
        if iterations >= max_iter as usize { break; }
        let mut next = theta.clone();
        for k in 0..2 {
            let (q, tr) = state.quadratic_trace[k];
            if use_mm {
                let tr_scale = 1.0 + tr.abs();
                let q_scale = 1.0 + q.abs();
                if !tr.is_finite() || tr <= 1e-12 * tr_scale {
                    return Err(fail(format!("MM denominator tr(P K_{}) is nonpositive or numerically zero.", k+1)));
                }
                if !q.is_finite() || q < -1e-10 * q_scale {
                    return Err(fail(format!("MM numerator for kernel {} is negative/nonfinite.", k+1)));
                }
                next[k] = theta[k] * (q.max(0.0) / tr).sqrt();
            } else {
                let rk = ranks[k] as f64;
                next[k] = ((theta[k] * (rk - theta[k] * tr) + theta[k] * theta[k] * q) / rk).max(0.0);
            }
            if !next[k].is_finite() { return Err(fail("Nonfinite spectral variance-component update.")); }
        }
        let mut candidate = next;
        let mut accepted: Option<(Vec<f64>, State)> = None;
        let slack = 1e-9 * (1.0 + state.loglik.abs());
        for _ in 0..=25 {
            if let Ok(ns) = evaluate(y, &x, d, &candidate, logdet_reference, restricted) {
                if ns.loglik + slack >= state.loglik {
                    accepted = Some((candidate, ns));
                    break;
                }
            }
            candidate = theta.iter().zip(candidate.iter())
                .map(|(a,b)| a + 0.5 * (b - a)).collect();
        }
        let (next, ns) = accepted.ok_or_else(|| fail(
            "Spectral update was not feasible/monotone after backtracking."
        ))?;
        relative_change = theta.iter().zip(next.iter())
            .map(|(a,b)| (a-b).abs() / (1.0+a.abs())).fold(0.0_f64, f64::max);
        relative_ll_change = (ns.loglik - state.loglik).abs() / (1.0 + state.loglik.abs());
        theta = next;
        state = ns;
        history.push(state.loglik);
        iterations += 1;
    }
    let param_ok = iterations > 0 && relative_change < tol;
    let kkt_ok = final_diag.scaled <= kkt_tol;
    Ok(list!(
        coefficients = state.beta,
        variance_components = theta,
        // Reconstructed on the original response scale in R/vcm.R.
        fitted_values = Vec::<f64>::new(),
        residuals = Vec::<f64>::new(),
        log_likelihood = state.loglik,
        loglik_history = history,
        converged = converged,
        iterations = iterations as i32,
        relative_change = relative_change,
        relative_loglik_change = relative_ll_change,
        score = state.score.to_vec(),
        kkt_violation = final_diag.raw,
        kkt_violation_scaled = final_diag.scaled,
        boundary_components = final_diag.on_boundary.to_vec(),
        converged_parameters = param_ok,
        converged_kkt = kkt_ok,
        convergence_reason = reason,
        stop_rule = stop_rule,
        kkt_tol = kkt_tol,
        boundary_tol = boundary_tol,
        pet = if restricted { "REML" } else { "MLE" },
        method = if use_mm { "MM" } else { "EM" }
    ))
}
