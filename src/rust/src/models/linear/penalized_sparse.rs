//! Native CSC Gaussian ridge/lasso/elastic-net, one lambda per call.
//!
//! Sparse predictors are never centered or converted to a dense matrix.
//! The intercept, if requested, is a synthetic unpenalized coordinate.
//! CSC arrays borrow their storage from R for the duration of this FFI call.
//! Criterion matches `penalized::ffi_fit`: weighted RSS/(2*sum(w)) plus
//! lambda * sum_j d_j(alpha*|beta_j|+(1-alpha)*beta_j^2/2).

use extendr_api::prelude::*;
use crate::core::{Pet, storage::{numeric_vector_or_matrix_from_robj,
    optional_numeric_vector_from_robj}};

#[inline]
fn soft_threshold(x: f64, threshold: f64) -> f64 {
    if x > threshold { x - threshold }
    else if x < -threshold { x + threshold }
    else { 0.0 }
}

#[allow(clippy::too_many_arguments)]
pub fn ffi_fit_sparse(
    y_obj: Robj,
    nrows: i32,
    col_ptr_obj: Robj,
    row_idx_obj: Robj,
    values_obj: Robj,
    weights_obj: Robj,
    offset_obj: Robj,
    penalty_obj: Robj,
    pet_name: &str,
    lambda: f64,
    alpha: f64,
    method_name: &str,
    intercept: bool,
    max_iter: i32,
    tol: f64,
) -> extendr_api::Result<List> {
    use extendr_api::{list, Error};
    let pet = Pet::from_regularized_str(pet_name, lambda, alpha)?;
    let (alpha, pet_label) = match pet {
        Pet::Ridge { .. } => (0.0, "Ridge"),
        Pet::Lasso { .. } => (1.0, "Lasso"),
        Pet::ElasticNet { alpha, .. } => (alpha, "ElasticNet"),
        _ => return Err(Error::Other("Sparse lmnet supports Ridge, Lasso, and ElasticNet.".into())),
    };
    if !matches!(method_name.trim().to_ascii_lowercase().replace('-', "_").as_str(),
        "auto" | "coordinate_descent" | "cd") {
        return Err(Error::Other(
            "Sparse lmnet supports only 'auto' or 'coordinate_descent'.".into()
        ));
    }
    if max_iter <= 0 || !tol.is_finite() || tol <= 0.0 {
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
    // Require canonical CSC: no duplicate or unsorted row indices per column.
    // Rejecting malformed input is essential for correct curvature and gradients.
    for j in 0..p {
        if cp[j] < 0 || cp[j+1] < cp[j] || cp[j+1] as usize > vx.len() {
            return Err(Error::Other(format!("Malformed CSC pointer at column {}.", j+1)));
        }
        let mut previous: Option<usize> = None;
        for k in cp[j] as usize .. cp[j+1] as usize {
            if ri[k] < 0 || (ri[k] as usize) >= n {
                return Err(Error::Other(format!("CSC row index out of bounds in column {}.", j+1)));
            }
            let row = ri[k] as usize;
            if previous.is_some_and(|prev| row <= prev) {
                return Err(Error::Other(format!("CSC row indices must be strictly increasing in column {}.", j+1)));
            }
            previous = Some(row);
            if !vx[k].is_finite() {
                return Err(Error::Other(format!("Non-finite CSC entry in column {}.", j+1)));
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
    let mut sum_w = 0.0;
    for i in 0..n {
        let w = weights[i];
        if !w.is_finite() || w < 0.0 || !offset[i].is_finite() {
            return Err(Error::Other(format!("Non-finite/negative weight or non-finite offset at row {}.", i+1)));
        }
        sum_w += w;
    }
    if !sum_w.is_finite() || sum_w <= 0.0 {
        return Err(Error::Other("Weights must have finite positive sum.".into()));
    }
    for (j, &d) in penalty.iter().enumerate() {
        if !d.is_finite() || d < 0.0 || !(d * lambda).is_finite() {
            return Err(Error::Other(format!("Invalid penalty.factor at column {}.", j+1)));
        }
    }
    for r in 0..q {
        for i in 0..n {
            if !y[(i,r)].is_finite() {return Err(Error::Other("y must be finite.".into()));}
        }
    }

    let mut curvature = vec![0.0_f64; p];
    for j in 0..p {
        let mut sq = 0.0;
        for k in cp[j] as usize .. cp[j+1] as usize {
            let row = ri[k] as usize;
            sq += weights[row] * vx[k] * vx[k];
        }
        curvature[j] = sq / sum_w;
        if !curvature[j].is_finite() {
            return Err(Error::Other("Non-finite sparse column curvature; rescale X.".into()));
        }
    }

    let nc = p + usize::from(intercept);
    let mut coef = vec![0.0_f64; nc * q];
    let mut fitted = vec![0.0_f64; n * q];
    let mut residuals = vec![0.0_f64; n * q];
    let mut all_converged = true;
    let mut most_sweeps = 0usize;
    let mut worst_kkt = 0.0_f64;
    let mut weighted_rss = 0.0_f64;
    let mut penalty_value = 0.0_f64;
    let start = usize::from(intercept);

    for r in 0..q {
        let mut beta = vec![0.0_f64; nc];
        let mut residual: Vec<f64> = (0..n).map(|i| y[(i,r)] - offset[i]).collect();
        let mut last_kkt = f64::INFINITY;
        let mut converged = false;
        for sweep in 1..=max_iter as usize {
            if intercept {
                let gradient = (0..n).map(|i| weights[i]*residual[i]).sum::<f64>() / sum_w;
                if !gradient.is_finite() {
                    return Err(Error::Other("Non-finite intercept gradient.".into()));
                }
                beta[0] += gradient;
                for v in &mut residual { *v -= gradient; }
            }
            for j in 0..p {
                let old = beta[start+j];
                let l1 = lambda * alpha * penalty[j];
                let l2 = lambda * (1.0-alpha) * penalty[j];
                let denom = curvature[j] + l2;
                if denom == 0.0 { continue; }
                let mut dot = 0.0;
                for k in cp[j] as usize .. cp[j+1] as usize {
                    let row = ri[k] as usize;
                    dot += weights[row] * vx[k] * residual[row];
                }
                let partial = dot / sum_w + curvature[j] * old;
                let next = soft_threshold(partial, l1) / denom;
                if !next.is_finite() {
                    return Err(Error::Other("Sparse coordinate descent produced a non-finite coefficient.".into()));
                }
                let delta = next - old;
                if delta != 0.0 {
                    beta[start+j] = next;
                    for k in cp[j] as usize .. cp[j+1] as usize {
                        let row = ri[k] as usize;
                        residual[row] -= vx[k] * delta;
                    }
                }
            }
            let mut kkt_max = 0.0_f64;
            let mut grad_scale = 0.0_f64;
            if intercept {
                let grad0 = (0..n).map(|i| weights[i]*residual[i]).sum::<f64>() / sum_w;
                kkt_max = grad0.abs();
                grad_scale = grad0.abs();
            }
            for j in 0..p {
                let mut dot = 0.0;
                for k in cp[j] as usize .. cp[j+1] as usize {
                    let row = ri[k] as usize;
                    dot += weights[row] * vx[k] * residual[row];
                }
                let grad = dot / sum_w;
                let bj = beta[start+j];
                let smooth = grad - lambda * (1.0-alpha) * penalty[j] * bj;
                let threshold = lambda * alpha * penalty[j];
                let violation = if bj > 0.0 { (smooth - threshold).abs() }
                                else if bj < 0.0 { (smooth + threshold).abs() }
                                else { (smooth.abs() - threshold).max(0.0) };
                grad_scale = grad_scale.max(grad.abs());
                kkt_max = kkt_max.max(violation);
            }
            if !kkt_max.is_finite() {
                return Err(Error::Other("Non-finite sparse KKT residual.".into()));
            }
            last_kkt = kkt_max;
            most_sweeps = most_sweeps.max(sweep);
            if kkt_max <= tol * (1.0+grad_scale) {
                converged = true;
                break;
            }
        }
        all_converged &= converged;
        worst_kkt = worst_kkt.max(last_kkt);
        coef[r*nc..(r+1)*nc].copy_from_slice(&beta);
        for i in 0..n {
            let pos = r*n+i;
            residuals[pos] = residual[i];
            fitted[pos] = y[(i,r)] - residual[i];
            weighted_rss += weights[i]*residual[i]*residual[i];
        }
        for j in 0..p {
            let b = beta[start+j];
            penalty_value += penalty[j] * (alpha*b.abs() + 0.5*(1.0-alpha)*b*b);
        }
    }
    let objective = weighted_rss/(2.0*sum_w) + lambda*penalty_value;
    if !objective.is_finite() {
        return Err(Error::Other("Sparse objective is non-finite.".into()));
    }
    Ok(list!(
        coefficients = coef,
        fitted_values = fitted,
        residuals = residuals,
        objective = objective,
        weighted_rss = weighted_rss,
        penalty_value = penalty_value,
        weight_sum = sum_w,
        method = "coordinate_descent",
        pet = pet_label,
        alpha = alpha,
        converged = all_converged,
        iterations = most_sweeps as i32,
        kkt_violation = worst_kkt,
        n_responses = q as i32,
        n_coefficients = nc as i32
    ))
}
