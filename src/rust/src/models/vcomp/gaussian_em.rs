//! Exact Gaussian variance-component ML/REML with latent-component EM.
//!
//! Sigma(theta) = sum_k theta_k K_k, K_k >= 0, rank(K_k) = r_k > 0.
//! Update: theta_k <- theta_k + theta_k^2/r_k *
//!          (y' P K_k P y - tr(P K_k)).
//! P = Sigma^-1 for ML scores (with GLS beta), or
//! P = Sigma^-1 - Sigma^-1 X (X' Sigma^-1 X)^-1 X' Sigma^-1 for REML.
//! These are genuine EM updates for latent N(0, theta_k K_k) components.
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

#[allow(clippy::too_many_arguments)]
pub fn ffi_fit_em(y_obj: Robj, x_obj: Robj, kernel_array: Robj,
                  kernel_ranks: Robj, theta_obj: Robj,
                  pet_str: &str, method_str: &str,
                  max_iter: i32, tol: f64) -> extendr_api::Result<List> {
    let pet = Pet::try_from(pet_str)?;
    let restricted = match pet {
        Pet::Reml => true,
        Pet::Mle => false,
        _ => return Err(fail("Gaussian VCM requires PET = 'MLE' or 'REML'.")),
    };
    let method = ApproximationMethod::try_from(method_str)?;
    if !matches!(method, ApproximationMethod::Optimization(NumericalOptimizer::EmAlgorithm)) {
        return Err(fail("This VCM endpoint implements only method = 'EM'; MM is reserved for Iteration 5B."));
    }
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
    let mut theta = theta_initial.to_vec();
    let mut state = evaluate(y, &x, &kernels, &theta, restricted)?;
    let mut likelihood_path = vec![state.loglik];
    let mut converged = false;
    let mut iterations = 0usize;
    let mut final_relative_change = f64::INFINITY;
    for iter in 1..=(max_iter as usize) {
        let mut next = theta.clone();
        for k in 0..m {
            let (q, tr) = component_score(&state, &kernels[k]);
            // Expected complete-data quadratic: theta*r - theta^2*tr + theta^2*q.
            let expectation = theta[k] * ((ranks[k] as f64) - theta[k] * tr)
                + theta[k] * theta[k] * q;
            // Tiny negative values caused by roundoff are projected to the boundary.
            next[k] = (expectation / (ranks[k] as f64)).max(0.0);
            if !next[k].is_finite() {
                return Err(fail(format!("EM update for kernel {} is not finite; rescale kernels.", k+1)));
            }
        }
        final_relative_change = theta.iter().zip(next.iter())
            .map(|(a,b)| (a-b).abs() / (1.0+a.abs())).fold(0.0_f64, f64::max);
        let next_state = evaluate(y, &x, &kernels, &next, restricted)?;
        let slack = 1e-9 * (1.0 + state.loglik.abs());
        if next_state.loglik + slack < state.loglik {
            return Err(fail("EM likelihood decreased beyond rounding tolerance; check kernel PSD/rank or conditioning."));
        }
        theta = next;
        state = next_state;
        likelihood_path.push(state.loglik);
        iterations = iter;
        if final_relative_change < tol { converged = true; break; }
    }
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
        pet = if restricted { "REML" } else { "MLE" },
        method = "EM"
    ))
}
