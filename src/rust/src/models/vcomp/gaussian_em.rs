//! Exact Gaussian variance-component ML/REML with EM and genuine MM updates.
//!
//! Sigma(theta) = sum_k theta_k K_k, K_k >= 0, rank(K_k) = r_k > 0.
//! EM: theta_k(new) = theta_k + theta_k^2 / rank(K_k) * (q_k - t_k).
//! MM: theta_k(new) = theta_k * sqrt(q_k / t_k).
//! q_k = (Sigma^-1 (y-X beta_GLS))' K_k (Sigma^-1 (y-X beta_GLS)).
//! t_k = tr(Sigma^-1 K_k) for profiled ML; tr(P K_k) for REML,
//! P = Sigma^-1 - Sigma^-1 X (X' Sigma^-1 X)^-1 X' Sigma^-1.
//! Both algorithms optimize the exact (restricted) Gaussian likelihood.
//! EM uses latent Gaussian components; MM is a distinct minorization-maximization
//! algorithm, not an EM step with altered convergence tolerance.
//! IMPORTANT: the *likelihood* uses profiled GLS beta in both ML and REML.
//! This initial dense implementation is O(n^3) per iteration and stores K.

use extendr_api::prelude::*;
use faer::{Mat, Side};
use faer::prelude::Solve;
use crate::core::{storage::dense_matrix_from_robj, Pet, ApproximationMethod};
use crate::core::method::NumericalOptimizer;

fn fail(message: impl Into<String>) -> Error { Error::Other(message.into()) }

fn logdet_from_llt(l: faer::MatRef<'_, f64>) -> extendr_api::Result<f64> {
    let mut ans = 0.0;
    for i in 0..l.nrows() {
        let d = l[(i, i)];
        if !d.is_finite() || d <= 0.0 {
            return Err(fail("Nonpositive/nonfinite Cholesky pivot; covariance must be positive definite."));
        }
        ans += 2.0 * d.ln();
    }
    Ok(ans)
}

struct State {
    loglik: f64,
    beta: Vec<f64>,
    fitted: Vec<f64>,
    residual: Vec<f64>,
    alpha: Vec<f64>, // Sigma^-1 (y - X beta) = P y
    projector: Mat<f64>, // ML: Sigma^-1; REML: P
}

fn evaluate(y: &[f64], x: &Mat<f64>, kernels: &[Mat<f64>],
            theta: &[f64], restricted: bool) -> extendr_api::Result<State> {
    let (n, p) = (y.len(), x.ncols());
    let mut sigma = Mat::<f64>::zeros(n, n);
    for (k, kernel) in kernels.iter().enumerate() {
        for j in 0..n {
            for i in 0..n {
                sigma[(i, j)] += theta[k] * kernel[(i, j)];
            }
        }
    }
    let chol = sigma.llt(Side::Lower).map_err(|_| fail(
        "Sigma(theta) is not positive definite. Include an identifiable positive residual kernel and positive starting components."
    ))?;
    let logdet_sigma = logdet_from_llt(chol.L())?;
    let inv_x = chol.solve(x);
    let y_mat = Mat::<f64>::from_fn(n, 1, |i, _| y[i]);
    let inv_y = chol.solve(&y_mat);
    let xt_inv_x = Mat::<f64>::from_fn(p, p, |a, b| {
        (0..n).map(|i| x[(i, a)] * inv_x[(i, b)]).sum()
    });
    let xt_inv_y = Mat::<f64>::from_fn(p, 1, |a, _| {
        (0..n).map(|i| x[(i, a)] * inv_y[(i, 0)]).sum()
    });
    let xchol = xt_inv_x.llt(Side::Lower).map_err(|_| fail(
        "X' Sigma^-1 X is not positive definite; the fixed-effects design must have full column rank."
    ))?;
    let logdet_x = if restricted { logdet_from_llt(xchol.L())? } else { 0.0 };
    let beta_mat = xchol.solve(&xt_inv_y);
    let beta = (0..p).map(|a| beta_mat[(a, 0)]).collect::<Vec<_>>();
    let mut fitted = vec![0.0; n];
    let mut residual = vec![0.0; n];
    for i in 0..n {
        fitted[i] = (0..p).map(|j| x[(i, j)] * beta[j]).sum();
        residual[i] = y[i] - fitted[i];
    }
    let r = Mat::<f64>::from_fn(n, 1, |i, _| residual[i]);
    let alpha_mat = chol.solve(&r);
    let alpha = (0..n).map(|i| alpha_mat[(i, 0)]).collect::<Vec<_>>();
    let quad: f64 = residual.iter().zip(alpha.iter()).map(|(a,b)| a*b).sum();
    let degrees = if restricted { n - p } else { n };
    let loglik = -0.5 * ((degrees as f64) * (2.0 * std::f64::consts::PI).ln()
        + logdet_sigma + logdet_x + quad);
    if !loglik.is_finite() { return Err(fail("Nonfinite Gaussian log likelihood.")); }

    let inv_sigma = chol.solve(&Mat::<f64>::identity(n, n));
    let projector = if restricted {
        let correction_rhs = inv_x.transpose().to_owned(); // p x n
        let correction = xchol.solve(&correction_rhs); // p x n
        Mat::<f64>::from_fn(n, n, |i, j| {
            inv_sigma[(i,j)] - (0..p).map(|a| inv_x[(i,a)] * correction[(a,j)]).sum::<f64>()
        })
    } else { inv_sigma };
    Ok(State { loglik, beta, fitted, residual, alpha, projector })
}

fn component_score(state: &State, kernel: &Mat<f64>) -> (f64, f64) {
    let n = state.alpha.len();
    let mut quadratic = 0.0;
    let mut trace = 0.0;
    for j in 0..n {
        let mut ka = 0.0;
        for i in 0..n {
            ka += kernel[(j, i)] * state.alpha[i];
            trace += state.projector[(i, j)] * kernel[(j, i)];
        }
        quadratic += state.alpha[j] * ka;
    }
    (quadratic, trace)
}

// The raw score is d log L / d theta_k = 0.5 * (q_k - t_k).
// At an interior KKT maximum it is zero; at theta_k == 0 it is <= 0.
// The scaled diagnostic divides by a term on the natural q/t scale,
// allowing kernels of very different magnitudes to be compared.
struct Diagnostics {
    quadratic_trace: Vec<(f64, f64)>,
    score: Vec<f64>,
    raw_kkt: f64,
    scaled_kkt: f64,
    on_boundary: Vec<bool>,
}

fn diagnostics(state: &State, kernels: &[Mat<f64>], theta: &[f64],
               boundary_tol: f64) -> Diagnostics {
    let theta_scale = theta.iter().copied().fold(0.0_f64, f64::max);
    let threshold = boundary_tol * (1.0 + theta_scale);
    let mut quadratic_trace = Vec::with_capacity(kernels.len());
    let mut score = Vec::with_capacity(kernels.len());
    let mut on_boundary = Vec::with_capacity(kernels.len());
    let mut raw_kkt = 0.0_f64;
    let mut scaled_kkt = 0.0_f64;
    for (k, kernel) in kernels.iter().enumerate() {
        let (q, tr) = component_score(state, kernel);
        let s = 0.5 * (q - tr);
        let boundary = theta[k] <= threshold;
        let violation = if boundary { s.max(0.0) } else { s.abs() };
        // Normalization reduces (but does not eliminate) sensitivity to kernel
    // scaling because the stabilizing +1 fixes a reference scale.
        let scaled = violation / (1.0 + 0.5 * (q.abs() + tr.abs()));
        quadratic_trace.push((q, tr));
        score.push(s);
        on_boundary.push(boundary);
        raw_kkt = raw_kkt.max(violation);
        scaled_kkt = scaled_kkt.max(scaled);
    }
    Diagnostics {quadratic_trace, score, raw_kkt, scaled_kkt, on_boundary}
}

#[allow(clippy::too_many_arguments)]
pub fn ffi_fit_em(y_obj: Robj, x_obj: Robj, kernel_array: Robj,
                  kernel_ranks: Robj, theta_obj: Robj,
                  pet_str: &str, method_str: &str,
                  max_iter: i32, tol: f64,
                  stop_rule_str: &str, kkt_tol: f64, boundary_tol: f64)
                  -> extendr_api::Result<List> {
    let pet = Pet::try_from(pet_str)?;
    let restricted = match pet {
        Pet::Reml => true,
        Pet::Mle => false,
        _ => return Err(fail("Gaussian VCM requires PET = 'MLE' or 'REML'.")),
    };
    let method = ApproximationMethod::try_from(method_str)?;
    let use_mm = match method {
        ApproximationMethod::Optimization(NumericalOptimizer::MmAlgorithm) => true,
        ApproximationMethod::Optimization(NumericalOptimizer::EmAlgorithm) => false,
        _ => return Err(fail("Gaussian VCM supports only method = 'EM' or 'MM'.")),
    };
    if max_iter <= 0 || !tol.is_finite() || tol <= 0.0 {
        return Err(fail("max_iter must be positive and tol must be finite and positive."));
    }
    let x = dense_matrix_from_robj(&x_obj, "X")?;
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 || (restricted && p >= n) {
        return Err(fail("X requires n > p >= 1 for REML; n and p must be positive for ML."));
    }
    let y = y_obj.as_real_slice().ok_or_else(|| fail("y must be a double vector."))?;
    if y.len() != n || y.iter().any(|v| !v.is_finite()) {
        return Err(fail("y must have n finite observations."));
    }
    let theta_initial = theta_obj.as_real_slice().ok_or_else(|| fail("theta_init must be a double vector."))?;
    let ranks = kernel_ranks.as_integer_slice().ok_or_else(|| fail("kernel ranks must be integer."))?;
    let kernel_data = kernel_array.as_real_slice().ok_or_else(|| fail("kernels must be a double 3D array."))?;
    let m = ranks.len();
    if m == 0 || theta_initial.len() != m ||
        n.checked_mul(n).and_then(|v| v.checked_mul(m)) != Some(kernel_data.len()) {
        return Err(fail("Kernel array dimensions, ranks, and theta_init disagree."));
    }
    if theta_initial.iter().any(|v| !v.is_finite() || *v <= 0.0) ||
       ranks.iter().any(|r| *r <= 0 || (*r as usize) > n) ||
       kernel_data.iter().any(|v| !v.is_finite()) {
        return Err(fail("Initial theta values must be positive and finite; kernels and positive ranks must be finite and valid."));
    }
    let kernels: Vec<Mat<f64>> = (0..m).map(|k| {
        Mat::<f64>::from_fn(n, n, |i,j| kernel_data[k*n*n + i + n*j])
    }).collect();
    // PSD/rank checks are performed on the R side using symmetric eigensystems.
    // Preserve the legacy parameter-change stopping criterion by default.
    // KKT diagnostics are evaluated using the exact likelihood score, not
    // surrogate gradients. No optimizer update is changed by this iteration.
    let stop_rule = match stop_rule_str.to_ascii_lowercase().as_str() {
        "parameter" => "parameter",
        "kkt" => "kkt",
        "both" => "both",
        _ => return Err(fail("stop_rule must be 'parameter', 'kkt', or 'both'.")),
    };
    if !kkt_tol.is_finite() || kkt_tol <= 0.0 ||
       !boundary_tol.is_finite() || boundary_tol < 0.0 {
        return Err(fail("kkt_tol must be positive and finite; boundary_tol must be nonnegative and finite."));
    }
    let mut theta = theta_initial.to_vec();
    let mut state = evaluate(y, &x, &kernels, &theta, restricted)?;
    let mut likelihood_path = vec![state.loglik];
    let mut converged = false;
    let mut iterations = 0usize;
    let mut final_relative_change = f64::INFINITY;
    let mut final_relative_likelihood_change = f64::INFINITY;
    let mut final_diag;
    let mut convergence_reason = "max_iter";

    // Diagnostics at the start of each iteration become the final diagnostics
    // without an additional likelihood factorization at termination.
    loop {
        final_diag = diagnostics(&state, &kernels, &theta, boundary_tol);
        if !final_diag.scaled_kkt.is_finite() {
            return Err(fail("Nonfinite variance-component score/KKT diagnostic."));
        }
        let param_ok = iterations > 0 && final_relative_change < tol;
        let kkt_ok = final_diag.scaled_kkt <= kkt_tol;
        converged = match stop_rule {
            "parameter" => param_ok,
            "kkt" => kkt_ok,
            _ => param_ok && kkt_ok,
        };
        if converged {
            convergence_reason = match stop_rule {
                "parameter" => "parameter",
                "kkt" => "kkt",
                _ => "both",
            };
            break;
        }
        if iterations >= max_iter as usize { break; }
        let mut next = theta.clone();
        for k in 0..m {
            let (q, tr) = final_diag.quadratic_trace[k];
            if use_mm {
                // Zhou et al. (2019), Equation (8): separable MM minorizer.
                // A zero projected REML kernel is unidentifiable and cannot
                // be updated by this multiplicative formula.
                let tr_scale = 1.0 + tr.abs();
                let q_scale = 1.0 + q.abs();
                if !tr.is_finite() || tr <= 1e-12 * tr_scale {
                    return Err(fail(format!(
                        "MM denominator tr(P K_{}) is nonpositive or numerically zero; the component may be unidentifiable (especially under REML).", k+1
                    )));
                }
                if !q.is_finite() || q < -1e-10 * q_scale {
                    return Err(fail(format!(
                        "MM numerator for kernel {} is negative or nonfinite; check PSD kernels and numerical conditioning.", k+1
                    )));
                }
                next[k] = theta[k] * (q.max(0.0) / tr).sqrt();
            } else {
                // Latent Gaussian EM M-step; retain the existing equation.
                let expectation = theta[k] * ((ranks[k] as f64) - theta[k] * tr)
                    + theta[k] * theta[k] * q;
                next[k] = (expectation / (ranks[k] as f64)).max(0.0);
            }
            if !next[k].is_finite() {
                return Err(fail(format!("{} update for kernel {} is not finite; rescale kernels.",
                    if use_mm { "MM" } else { "EM" }, k+1)));
            }
        }
        // A proposed boundary update can make Sigma singular. Backtrack
        // toward the previous feasible iterate, also safeguarding against
        // loss of monotonicity from finite precision. This is inactive when
        // the original EM/MM step is feasible and ascends.
        let mut candidate = next;
        let mut accepted: Option<(Vec<f64>, State)> = None;
        let slack = 1e-9 * (1.0 + state.loglik.abs());
        for _ in 0..=25 {
            if let Ok(next_state) = evaluate(y, &x, &kernels, &candidate, restricted) {
                if next_state.loglik + slack >= state.loglik {
                    accepted = Some((candidate, next_state));
                    break;
                }
            }
            candidate = theta.iter().zip(candidate.iter())
                .map(|(old, proposed)| old + 0.5 * (proposed - old))
                .collect();
        }
        let (next, next_state) = accepted.ok_or_else(|| fail(
            "VCM step was not feasible/monotone after backtracking; inspect kernel identifiability and conditioning."
        ))?;
        final_relative_change = theta.iter().zip(next.iter())
            .map(|(a,b)| (a-b).abs() / (1.0+a.abs())).fold(0.0_f64, f64::max);
        final_relative_likelihood_change = (next_state.loglik - state.loglik).abs()
            / (1.0 + state.loglik.abs());
        theta = next;
        state = next_state;
        likelihood_path.push(state.loglik);
        iterations += 1;
    }
    let param_ok = iterations > 0 && final_relative_change < tol;
    let kkt_ok = final_diag.scaled_kkt <= kkt_tol;
    Ok(list!(
        coefficients = state.beta,
        variance_components = theta,
        fitted_values = state.fitted,
        residuals = state.residual,
        log_likelihood = state.loglik,
        loglik_history = likelihood_path,
        converged = converged,
        iterations = iterations as i32,
        relative_change = final_relative_change,
        relative_loglik_change = final_relative_likelihood_change,
        score = final_diag.score,
        kkt_violation = final_diag.raw_kkt,
        kkt_violation_scaled = final_diag.scaled_kkt,
        boundary_components = final_diag.on_boundary,
        converged_parameters = param_ok,
        converged_kkt = kkt_ok,
        convergence_reason = convergence_reason,
        stop_rule = stop_rule,
        kkt_tol = kkt_tol,
        boundary_tol = boundary_tol,
        pet = if restricted { "REML" } else { "MLE" },
        method = if use_mm { "MM" } else { "EM" }
    ))
}
