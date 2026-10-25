//! Native CSC Gaussian regularization paths with warm-start coordinate descent.
//!
//! No dense design or Gram matrix is constructed. The CSC slots are borrowed
//! for the FFI call, and coefficients/residuals are carried between decreasing
//! lambda values. Output coefficient layout is [lambda][response][coefficient],
//! matching R's column-major array dimensions (ncoef, nresponse, nlambda).

use extendr_api::prelude::*;
use crate::core::{Pet, storage::{numeric_vector_or_matrix_from_robj,
    optional_numeric_vector_from_robj}};

#[inline]
fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold { value - threshold }
    else if value < -threshold { value + threshold }
    else { 0.0 }
}

#[allow(clippy::too_many_arguments)]
pub fn ffi_fit_sparse_path(
    y_obj: Robj,
    nrows: i32,
    col_ptr_obj: Robj,
    row_idx_obj: Robj,
    values_obj: Robj,
    weights_obj: Robj,
    offset_obj: Robj,
    penalty_obj: Robj,
    pet_name: &str,
    lambda_obj: Robj,
    alpha: f64,
    method_name: &str,
    intercept: bool,
    max_iter: i32,
    tol: f64,
) -> extendr_api::Result<List> {
    use extendr_api::{list, Error};

    let lambdas = lambda_obj.as_real_slice().ok_or_else(|| Error::Other(
        "lambda must be a numeric double vector.".into()
    ))?;
    if lambdas.len() < 2 {
        return Err(Error::Other("A sparse path needs two or more lambda values.".into()));
    }
    for (k, &lam) in lambdas.iter().enumerate() {
        if !lam.is_finite() || lam < 0.0 {
            return Err(Error::Other(format!("lambda[{}] must be finite and nonnegative.", k + 1)));
        }
        if k > 0 && lam >= lambdas[k - 1] {
            return Err(Error::Other("lambda must be strictly decreasing.".into()));
        }
    }
    let pet = Pet::from_regularized_str(pet_name, lambdas[0], alpha)?;
    let (alpha, pet_label) = match pet {
        Pet::Ridge { .. } => (0.0, "Ridge"),
        Pet::Lasso { .. } => (1.0, "Lasso"),
        Pet::ElasticNet { alpha, .. } => (alpha, "ElasticNet"),
        _ => return Err(Error::Other("Unsupported sparse penalized PET.".into())),
    };
    if !matches!(method_name.trim().to_ascii_lowercase().replace('-', "_").as_str(),
        "auto" | "coordinate_descent" | "cd") {
        return Err(Error::Other(
            "Sparse paths support 'auto' or 'coordinate_descent' only.".into()
        ));
    }
    if max_iter < 1 || !tol.is_finite() || tol <= 0.0 {
        return Err(Error::Other("max_iter must be positive and tol must be finite and positive.".into()));
    }
    if nrows <= 0 { return Err(Error::Other("nrows must be positive.".into())); }
    let n = nrows as usize;

    let cp = col_ptr_obj.as_integer_slice().ok_or_else(|| Error::Other(
        "CSC column pointers must be an integer vector.".into()
    ))?;
    let ri = row_idx_obj.as_integer_slice().ok_or_else(|| Error::Other(
        "CSC row indices must be an integer vector.".into()
    ))?;
    let vx = values_obj.as_real_slice().ok_or_else(|| Error::Other(
        "CSC nonzero values must be a double vector.".into()
    ))?;
    if cp.len() < 2 { return Err(Error::Other("CSC needs at least one column.".into())); }
    let p = cp.len() - 1;
    if ri.len() != vx.len() || cp[0] != 0 || cp[p] < 0 || cp[p] as usize != vx.len() {
        return Err(Error::Other("Malformed CSC pointers or nonzero arrays.".into()));
    }
    // Canonical CSC has sorted, unique row indices in each column.
    for j in 0..p {
        if cp[j] < 0 || cp[j + 1] < cp[j] || cp[j + 1] as usize > vx.len() {
            return Err(Error::Other(format!("Malformed CSC pointer at column {}.", j + 1)));
        }
        let mut previous: Option<usize> = None;
        for k in cp[j] as usize .. cp[j + 1] as usize {
            if ri[k] < 0 || ri[k] as usize >= n {
                return Err(Error::Other(format!("CSC row index out of bounds in column {}.", j + 1)));
            }
            let row = ri[k] as usize;
            if previous.is_some_and(|prev| row <= prev) {
                return Err(Error::Other(format!("CSC row indices must be strictly increasing in column {}.", j + 1)));
            }
            previous = Some(row);
            if !vx[k].is_finite() {
                return Err(Error::Other(format!("Non-finite CSC entry in column {}.", j + 1)));
            }
        }
    }

    let y = numeric_vector_or_matrix_from_robj(&y_obj, "y", 1)?;
    if y.nrows() != n || y.ncols() == 0 {
        return Err(Error::Other("y must have nrows observations and at least one response.".into()));
    }
    let q = y.ncols();
    let weights = optional_numeric_vector_from_robj(&weights_obj, n, 1.0, "weights")?;
    let offset = optional_numeric_vector_from_robj(&offset_obj, n, 0.0, "offset")?;
    let penalty = optional_numeric_vector_from_robj(&penalty_obj, p, 1.0, "penalty.factor")?;
    let mut sum_w = 0.0_f64;
    for i in 0..n {
        if !weights[i].is_finite() || weights[i] < 0.0 || !offset[i].is_finite() {
            return Err(Error::Other(format!("Non-finite/negative weight or non-finite offset at row {}.", i + 1)));
        }
        sum_w += weights[i];
    }
    if !sum_w.is_finite() || sum_w <= 0.0 {
        return Err(Error::Other("Weights must have finite positive sum.".into()));
    }
    for (j, &d) in penalty.iter().enumerate() {
        if !d.is_finite() || d < 0.0 || !(d * lambdas[0]).is_finite() {
            return Err(Error::Other(format!("Invalid penalty.factor at column {}.", j + 1)));
        }
    }
    for r in 0..q {
        for i in 0..n {
            if !y[(i, r)].is_finite() {
                return Err(Error::Other("y must be finite.".into()));
            }
        }
    }
    let mut curvature = vec![0.0_f64; p];
    for j in 0..p {
        let mut sq = 0.0;
        for k in cp[j] as usize .. cp[j + 1] as usize {
            sq += weights[ri[k] as usize] * vx[k] * vx[k];
        }
        curvature[j] = sq / sum_w;
        if !curvature[j].is_finite() {
            return Err(Error::Other("Non-finite sparse column curvature; rescale X.".into()));
        }
    }

    let nlambda = lambdas.len();
    let nc = p + usize::from(intercept);
    let coef_count = nc.checked_mul(q).and_then(|v| v.checked_mul(nlambda)).ok_or_else(||
        Error::Other("Coefficient path dimension overflow.".into())
    )?;
    let mut coefficient_path = vec![0.0_f64; coef_count];
    let mut objective = vec![0.0_f64; nlambda];
    let mut weighted_rss = vec![0.0_f64; nlambda];
    let mut penalty_value = vec![0.0_f64; nlambda];
    let mut iterations = vec![0_i32; nlambda];
    let mut converged = vec![true; nlambda];
    let mut kkt_violation = vec![0.0_f64; nlambda];
    let start = usize::from(intercept);

    for r in 0..q {
        let mut beta = vec![0.0_f64; nc];
        let mut residual: Vec<f64> = (0..n).map(|i| y[(i, r)] - offset[i]).collect();

        // Reuse both coefficient and residual buffers for each subsequent lambda.
        for (lambda_idx, &lambda) in lambdas.iter().enumerate() {
            let mut found = false;
            let mut last_kkt = f64::INFINITY;
            for sweep in 1..=max_iter as usize {
                if intercept {
                    let gradient = (0..n).map(|i| weights[i] * residual[i]).sum::<f64>() / sum_w;
                    if !gradient.is_finite() {
                        return Err(Error::Other("Non-finite intercept gradient.".into()));
                    }
                    beta[0] += gradient;
                    for v in &mut residual { *v -= gradient; }
                }
                for j in 0..p {
                    let old = beta[start + j];
                    let l1 = lambda * alpha * penalty[j];
                    let l2 = lambda * (1.0 - alpha) * penalty[j];
                    let denom = curvature[j] + l2;
                    if denom == 0.0 { continue; }
                    let mut dot = 0.0;
                    for k in cp[j] as usize .. cp[j + 1] as usize {
                        dot += weights[ri[k] as usize] * vx[k] * residual[ri[k] as usize];
                    }
                    let partial = dot / sum_w + curvature[j] * old;
                    let next = soft_threshold(partial, l1) / denom;
                    if !next.is_finite() {
                        return Err(Error::Other("Sparse path produced a non-finite coefficient.".into()));
                    }
                    let delta = next - old;
                    if delta != 0.0 {
                        beta[start + j] = next;
                        for k in cp[j] as usize .. cp[j + 1] as usize {
                            residual[ri[k] as usize] -= vx[k] * delta;
                        }
                    }
                }
                let mut kkt_max = 0.0_f64;
                let mut grad_scale = 0.0_f64;
                if intercept {
                    let grad0 = (0..n).map(|i| weights[i] * residual[i]).sum::<f64>() / sum_w;
                    kkt_max = grad0.abs();
                    grad_scale = grad0.abs();
                }
                for j in 0..p {
                    let mut dot = 0.0;
                    for k in cp[j] as usize .. cp[j + 1] as usize {
                        dot += weights[ri[k] as usize] * vx[k] * residual[ri[k] as usize];
                    }
                    let grad = dot / sum_w;
                    let bj = beta[start + j];
                    let smooth = grad - lambda * (1.0 - alpha) * penalty[j] * bj;
                    let cutoff = lambda * alpha * penalty[j];
                    let violation = if bj > 0.0 { (smooth - cutoff).abs() }
                        else if bj < 0.0 { (smooth + cutoff).abs() }
                        else { (smooth.abs() - cutoff).max(0.0) };
                    grad_scale = grad_scale.max(grad.abs());
                    kkt_max = kkt_max.max(violation);
                }
                if !kkt_max.is_finite() {
                    return Err(Error::Other("Non-finite sparse path KKT residual.".into()));
                }
                last_kkt = kkt_max;
                iterations[lambda_idx] = iterations[lambda_idx].max(sweep as i32);
                if kkt_max <= tol * (1.0 + grad_scale) {
                    found = true;
                    break;
                }
            }
            converged[lambda_idx] &= found;
            kkt_violation[lambda_idx] = kkt_violation[lambda_idx].max(last_kkt);
            let path_start = lambda_idx * (nc * q) + r * nc;
            coefficient_path[path_start .. path_start + nc].copy_from_slice(&beta);
            for j in 0..p {
                let b = beta[start + j];
                penalty_value[lambda_idx] += penalty[j] *
                    (alpha * b.abs() + 0.5 * (1.0 - alpha) * b * b);
            }
            for i in 0..n {
                weighted_rss[lambda_idx] += weights[i] * residual[i] * residual[i];
            }
        }
    }
    for (k, &lam) in lambdas.iter().enumerate() {
        objective[k] = weighted_rss[k] / (2.0 * sum_w) + lam * penalty_value[k];
        if !objective[k].is_finite() {
            return Err(Error::Other("Non-finite sparse path objective; rescale inputs.".into()));
        }
    }
    Ok(list!(
        coefficients = coefficient_path,
        lambda = lambdas.to_vec(),
        alpha = alpha,
        pet = pet_label,
        method = "coordinate_descent",
        objective = objective,
        weighted_rss = weighted_rss,
        penalty_value = penalty_value,
        iterations = iterations,
        converged = converged,
        kkt_violation = kkt_violation,
        weight_sum = sum_w,
        n_coefficients = nc as i32,
        n_responses = q as i32,
        n_lambda = nlambda as i32
    ))
}
