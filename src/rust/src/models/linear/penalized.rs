//! Gaussian penalized linear regression: Ridge, Lasso, ElasticNet.
//!
//! Criterion, with S = sum_i w_i and q responses:
//!   (1 / (2 S)) sum_{i,r} w_i (y_ir - offset_i - x_i' beta_r)^2
//!       + lambda * sum_{j,r} penalty_factor_j *
//!           (alpha * abs(beta_jr) + (1-alpha)/2 * beta_jr^2).
//!
//! A nonpenalized intercept corresponds to penalty_factor_j = 0.
//! No automatic standardization is applied.  The coefficient order is the
//! original (unpivoted) order and multiple responses share the factorization.

use extendr_api::prelude::*;
use faer::{Mat, MatRef, Side};
use faer::prelude::{Solve, SolveLstsq};
use crate::core::{
    method::{ClosedFormSolver, NumericalOptimizer},
    storage::{InputMatrix, numeric_vector_or_matrix_from_robj,
              optional_numeric_vector_from_robj},
    ApproximationMethod, Pet,
};

fn checked_dense_ref<'a>(obj: &'a Robj, role: &str) -> extendr_api::Result<MatRef<'a, f64>> {
    use extendr_api::Error;
    match InputMatrix::try_from_robj(obj)? {
        InputMatrix::Dense(mat) => Ok(mat),
        InputMatrix::SparseCscOwned { .. } => Err(Error::Other(format!(
            "{role}: sparse fitting is not yet implemented; refusing to densify implicitly."
        ))),
        InputMatrix::CustomPlaceholder(_) => Err(Error::Other(format!(
            "{role} must be a numeric double matrix."
        ))),
    }
}

fn vec_col_major(m: &Mat<f64>) -> Vec<f64> {
    let mut v = Vec::with_capacity(m.nrows() * m.ncols());
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            v.push(m[(i, j)]);
        }
    }
    v
}

// Residual-updating cyclic coordinate descent. X is *borrowed* directly
// from the R numeric matrix; no Gram matrix or additional n-by-p copy.
// Response columns are optimized separately because this PET has a separable
// per-response penalty, not a group/multitask penalty.
fn coordinate_descent(
    x: &MatRef<'_, f64>,
    y: &Mat<f64>,
    weights: &[f64],
    offset: &[f64],
    penalty: &[f64],
    lambda: f64,
    alpha: f64,
    sum_w: f64,
    max_iter: usize,
    tol: f64,
) -> extendr_api::Result<(Mat<f64>, usize, bool, f64)> {
    use extendr_api::Error;
    let (n, p, q) = (x.nrows(), x.ncols(), y.ncols());
    let mut col_curvature = vec![0.0_f64; p];
    for j in 0..p {
        let mut d = 0.0;
        for i in 0..n {
            d += weights[i] * x[(i, j)] * x[(i, j)];
        }
        col_curvature[j] = d / sum_w;
        if !col_curvature[j].is_finite() {
            return Err(Error::Other("Non-finite weighted column norm; rescale X.".into()));
        }
    }

    let mut beta = Mat::<f64>::zeros(p, q);
    let mut all_converged = true;
    let mut max_iterations = 0_usize;
    let mut max_violation = 0.0_f64;
    for r in 0..q {
        // r = Y - offset - X beta; beta starts at zero.
        let mut residual = (0..n).map(|i| y[(i, r)] - offset[i]).collect::<Vec<_>>();
        let mut converged = false;
        let mut last_violation = f64::INFINITY;

        for sweep in 1..=max_iter {
            for j in 0..p {
                let old = beta[(j, r)];
                let l1 = lambda * alpha * penalty[j];
                let l2 = lambda * (1.0 - alpha) * penalty[j];
                let denom = col_curvature[j] + l2;
                if denom == 0.0 {
                    // A completely unidentifiable, unpenalized zero column.
                    // Our deterministic convention sets its coefficient to zero.
                    continue;
                }
                let mut grad = 0.0;
                for i in 0..n {
                    grad += weights[i] * x[(i, j)] * residual[i];
                }
                // Partial residual inner product at beta_j = 0.
                let partial = (grad / sum_w) + col_curvature[j] * old;
                let shrunk = if partial > l1 {
                    partial - l1
                } else if partial < -l1 {
                    partial + l1
                } else {
                    0.0
                };
                let next = shrunk / denom;
                if !next.is_finite() {
                    return Err(Error::Other("Coordinate descent generated non-finite coefficients; rescale the inputs.".into()));
                }
                let delta = next - old;
                if delta != 0.0 {
                    beta[(j, r)] = next;
                    for i in 0..n {
                        residual[i] -= x[(i, j)] * delta;
                    }
                }
            }

            // First-order optimality check, including subgradient at zero.
            // For alpha=0 this reduces to the weighted ridge normal equations.
            let mut kkt_max = 0.0_f64;
            let mut grad_scale = 0.0_f64;
            for j in 0..p {
                let mut grad = 0.0_f64;
                for i in 0..n {
                    grad += weights[i] * x[(i, j)] * residual[i];
                }
                grad /= sum_w;
                let bj = beta[(j, r)];
                let smooth = grad - lambda * (1.0 - alpha) * penalty[j] * bj;
                let threshold = lambda * alpha * penalty[j];
                let violation = if bj > 0.0 {
                    (smooth - threshold).abs()
                } else if bj < 0.0 {
                    (smooth + threshold).abs()
                } else {
                    (smooth.abs() - threshold).max(0.0)
                };
                grad_scale = grad_scale.max(grad.abs());
                kkt_max = kkt_max.max(violation);
            }
            if !kkt_max.is_finite() {
                return Err(Error::Other("Non-finite KKT violation; rescale the inputs.".into()));
            }
            last_violation = kkt_max;
            max_iterations = max_iterations.max(sweep);
            if kkt_max <= tol * (1.0 + grad_scale) {
                converged = true;
                break;
            }
        }
        all_converged &= converged;
        max_violation = max_violation.max(last_violation);
    }
    Ok((beta, max_iterations, all_converged, max_violation))
}

/// The input R design matrix is borrowed through faer::MatRef for the duration
/// of the synchronous call.  CD uses O(n + p + pq) scratch space (excluding
/// the returned fitted/residual arrays). QR/Cholesky retain their own workspace.
pub fn ffi_fit(
    y_obj: Robj,
    x_obj: Robj,
    weights_obj: Robj,
    offset_obj: Robj,
    penalty_factor_obj: Robj,
    pet_str: &str,
    lambda: f64,
    alpha: f64,
    method_str: &str,
    max_iter: i32,
    tol: f64,
) -> extendr_api::Result<List> {
    use extendr_api::{list, Error};
    let pet = Pet::from_regularized_str(pet_str, lambda, alpha)?;
    let (lambda, alpha, pet_label) = match pet {
        Pet::Ridge { lambda } => (lambda, 0.0_f64, "Ridge"),
        Pet::Lasso { lambda } => (lambda, 1.0_f64, "Lasso"),
        Pet::ElasticNet { lambda, alpha } => (lambda, alpha, "ElasticNet"),
        _ => return Err(Error::Other("Unsupported PET for Gaussian lmnet.".into())),
    };
    if max_iter < 1 {
        return Err(Error::Other("max_iter must be a positive integer.".into()));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(Error::Other("tol must be finite and positive.".into()));
    }

    let solver = if method_str.trim().eq_ignore_ascii_case("auto") {
        if matches!(pet, Pet::Ridge { .. }) { "cholesky" } else { "coordinate_descent" }
    } else {
        let method = ApproximationMethod::try_from(method_str)?;
        match method {
            ApproximationMethod::ClosedForm(ClosedFormSolver::Cholesky) => "cholesky",
            ApproximationMethod::ClosedForm(ClosedFormSolver::ColPivQr) => "qr",
            ApproximationMethod::ClosedForm(ClosedFormSolver::QrHouseholder) => "householder_qr",
            ApproximationMethod::Optimization(NumericalOptimizer::CoordinateDescent) => "coordinate_descent",
            _ => return Err(Error::Other("lmnet supports 'auto', 'cholesky', 'qr', 'householder_qr', or 'coordinate_descent'.".into())),
        }
    };
    if !matches!(pet, Pet::Ridge { .. }) && solver != "coordinate_descent" {
        return Err(Error::Other("Lasso and ElasticNet require method = 'coordinate_descent' (or 'auto').".into()));
    }

    let x = checked_dense_ref(&x_obj, "x")?;
    let y = numeric_vector_or_matrix_from_robj(&y_obj, "y", 1)?;
    let n = x.nrows();
    let p = x.ncols();
    let q = y.ncols();
    if n == 0 || p == 0 || q == 0 || y.nrows() != n {
        return Err(Error::Other("x and y must have compatible positive dimensions.".to_string()));
    }
    let weights = optional_numeric_vector_from_robj(&weights_obj, n, 1.0, "weights")?;
    let offset = optional_numeric_vector_from_robj(&offset_obj, n, 0.0, "offset")?;
    let penalty = optional_numeric_vector_from_robj(&penalty_factor_obj, p, 1.0, "penalty.factor")?;
    let mut sum_w = 0.0_f64;
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() || w < 0.0 {
            return Err(Error::Other(format!("weights must be finite and nonnegative (row {}).", i + 1)));
        }
        sum_w += w;
        if !offset[i].is_finite() {
            return Err(Error::Other(format!("offset must be finite (row {}).", i + 1)));
        }
    }
    if !sum_w.is_finite() || sum_w <= 0.0 {
        return Err(Error::Other("sum of observation weights must be positive and finite.".to_string()));
    }
    for (j, &d) in penalty.iter().enumerate() {
        if !d.is_finite() || d < 0.0 {
            return Err(Error::Other(format!("penalty.factor must be finite and nonnegative (column {}).", j + 1)));
        }
        if !(lambda * d).is_finite() {
            return Err(Error::Other(format!(
                "lambda times penalty.factor overflows at column {}; rescale the penalty.", j + 1
            )));
        }
    }
    for j in 0..p {
        for i in 0..n {
            if !x[(i,j)].is_finite() {
                return Err(Error::Other("x contains non-finite values.".to_string()));
            }
        }
    }
    for r in 0..q {
        for i in 0..n {
            if !y[(i,r)].is_finite() {
                return Err(Error::Other("y contains non-finite values.".to_string()));
            }
        }
    }

    // Weighted loss divided by S = sum(w), invariant when all weights are
    // multiplied by the same constant. Closed-form ridge uses a single dense
    // working matrix; CD borrows X directly and doesn't allocate n*p scratch.
    let (beta, iterations, converged, kkt_violation) = if solver == "coordinate_descent" {
        coordinate_descent(&x, &y, &weights, &offset, &penalty, lambda, alpha,
                           sum_w, max_iter as usize, tol)?
    } else {
        let extra = if solver == "cholesky" { 0 } else { p };
        let mut x_work = Mat::<f64>::zeros(n + extra, p);
        let mut y_work = Mat::<f64>::zeros(n + extra, q);
        for i in 0..n {
            let scale = (weights[i] / sum_w).sqrt();
            for j in 0..p {
                x_work[(i,j)] = x[(i,j)] * scale;
            }
            for r in 0..q {
                y_work[(i,r)] = (y[(i,r)] - offset[i]) * scale;
            }
        }
        if extra > 0 {
            for j in 0..p {
                x_work[(n+j,j)] = (lambda * penalty[j]).sqrt();
            }
        }
        let beta = if solver == "cholesky" {
            let mut gram = x_work.transpose() * &x_work;
            for j in 0..p {
                gram[(j,j)] += lambda * penalty[j];
            }
            let rhs = x_work.transpose() * &y_work;
            gram.llt(Side::Lower).map_err(|_| Error::Other(
                "Ridge Cholesky factorization failed; unpenalized directions may be rank deficient. Try method = 'qr' or use positive penalties.".to_string()
            ))?.solve(&rhs)
        } else if solver == "qr" {
            x_work.col_piv_qr().solve_lstsq(&y_work)
        } else {
            x_work.qr().solve_lstsq(&y_work)
        };
        (beta, 0_usize, true, 0.0_f64)
    };

    let mut fitted = Mat::<f64>::zeros(n,q);
    let mut resid = Mat::<f64>::zeros(n,q);
    let mut weighted_rss = 0.0_f64;
    for r in 0..q {
        for i in 0..n {
            let mut yh = offset[i];
            for j in 0..p {
                yh += x[(i,j)] * beta[(j,r)];
            }
            fitted[(i,r)] = yh;
            let e = y[(i,r)] - yh;
            resid[(i,r)] = e;
            weighted_rss += weights[i] * e * e;
        }
    }
    let mut penalty_value = 0.0_f64;
    for r in 0..q {
        for j in 0..p {
            let bj = beta[(j, r)];
            penalty_value += penalty[j] * (alpha * bj.abs() + 0.5 * (1.0 - alpha) * bj * bj);
        }
    }
    let objective = weighted_rss / (2.0 * sum_w) + lambda * penalty_value;
    if !objective.is_finite() {
        return Err(Error::Other("Penalized Gaussian fit produced a non-finite objective.".into()));
    }
    Ok(list!(
        coefficients = vec_col_major(&beta),
        fitted_values = vec_col_major(&fitted),
        residuals = vec_col_major(&resid),
        objective = objective,
        weighted_rss = weighted_rss,
        penalty_value = penalty_value,
        weight_sum = sum_w,
        method = solver,
        pet = pet_label,
        alpha = alpha,
        converged = converged,
        iterations = iterations as i32,
        kkt_violation = kkt_violation,
        n_responses = q as i32,
        n_coefficients = p as i32
    ))
}

/// Fit a strictly decreasing regularization path using one borrowed dense
/// design matrix and residual-updating cyclic coordinate descent. Coefficients
/// are stored in [lambda][response][coefficient] column-major order.
///
/// All lambdas share the PET variant and its fixed alpha; each lambda is a
/// distinct statistical target. Warm starts affect only numerical initialization.
pub fn ffi_fit_path(
    y_obj: Robj,
    x_obj: Robj,
    weights_obj: Robj,
    offset_obj: Robj,
    penalty_factor_obj: Robj,
    pet_str: &str,
    lambda_obj: Robj,
    alpha: f64,
    method_str: &str,
    max_iter: i32,
    tol: f64,
) -> extendr_api::Result<List> {
    use extendr_api::{list, Error};

    let lambdas = lambda_obj.as_real_slice().ok_or_else(|| Error::Other(
        "lambda must be a numeric double vector.".to_string()
    ))?;
    if lambdas.len() < 2 {
        return Err(Error::Other(
            "A regularization path needs at least two lambda values; use the scalar interface otherwise.".into()
        ));
    }
    for (k, &lam) in lambdas.iter().enumerate() {
        if !lam.is_finite() || lam < 0.0 {
            return Err(Error::Other(format!("lambda[{}] must be finite and nonnegative.", k + 1)));
        }
        if k > 0 && lam >= lambdas[k - 1] {
            return Err(Error::Other("lambda must be strictly decreasing for warm starts.".into()));
        }
    }
    let pet = Pet::from_regularized_str(pet_str, lambdas[0], alpha)?;
    let (alpha, pet_label) = match pet {
        Pet::Ridge { .. } => (0.0, "Ridge"),
        Pet::Lasso { .. } => (1.0, "Lasso"),
        Pet::ElasticNet { alpha, .. } => (alpha, "ElasticNet"),
        _ => return Err(Error::Other("Unsupported Gaussian penalized PET.".into())),
    };
    let method = if method_str.trim().eq_ignore_ascii_case("auto") {
        ApproximationMethod::Optimization(NumericalOptimizer::CoordinateDescent)
    } else {
        ApproximationMethod::try_from(method_str)?
    };
    if !matches!(method, ApproximationMethod::Optimization(NumericalOptimizer::CoordinateDescent)) {
        return Err(Error::Other(
            "Regularization paths support method = 'auto' or 'coordinate_descent' only; direct ridge methods fit one lambda at a time.".into()
        ));
    }
    if max_iter < 1 || !tol.is_finite() || tol <= 0.0 {
        return Err(Error::Other("max_iter must be positive and tol must be finite and positive.".into()));
    }

    // Borrow X instead of constructing an n-by-p copy at each lambda.
    let x = checked_dense_ref(&x_obj, "x")?;
    let y = numeric_vector_or_matrix_from_robj(&y_obj, "y", 1)?;
    let (n, p, q, nlambda) = (x.nrows(), x.ncols(), y.ncols(), lambdas.len());
    if n == 0 || p == 0 || q == 0 || y.nrows() != n {
        return Err(Error::Other("x and y must have compatible positive dimensions.".into()));
    }
    let weights = optional_numeric_vector_from_robj(&weights_obj, n, 1.0, "weights")?;
    let offset = optional_numeric_vector_from_robj(&offset_obj, n, 0.0, "offset")?;
    let penalty = optional_numeric_vector_from_robj(&penalty_factor_obj, p, 1.0, "penalty.factor")?;
    let mut sum_w = 0.0_f64;
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() || w < 0.0 {
            return Err(Error::Other(format!("weights must be finite and nonnegative (row {}).", i + 1)));
        }
        sum_w += w;
        if !offset[i].is_finite() {
            return Err(Error::Other(format!("offset must be finite (row {}).", i + 1)));
        }
    }
    if !sum_w.is_finite() || sum_w <= 0.0 {
        return Err(Error::Other("sum of observation weights must be positive and finite.".into()));
    }
    for (j, &d) in penalty.iter().enumerate() {
        if !d.is_finite() || d < 0.0 || !(lambdas[0] * d).is_finite() {
            return Err(Error::Other(format!("penalty.factor must be finite and nonnegative with finite lambda product (column {}).", j + 1)));
        }
    }
    for j in 0..p {
        for i in 0..n {
            if !x[(i, j)].is_finite() {
                return Err(Error::Other("x contains non-finite values.".into()));
            }
        }
    }
    for r in 0..q {
        for i in 0..n {
            if !y[(i, r)].is_finite() {
                return Err(Error::Other("y contains non-finite values.".into()));
            }
        }
    }

    // Coordinate curvature is lambda-independent, computed just once.
    let mut curvature = vec![0.0_f64; p];
    for j in 0..p {
        let mut d = 0.0;
        for i in 0..n {
            d += weights[i] * x[(i, j)] * x[(i, j)];
        }
        curvature[j] = d / sum_w;
        if !curvature[j].is_finite() {
            return Err(Error::Other("Non-finite weighted column norm; rescale X.".into()));
        }
    }

    // Only O(p*q*nlambda) coefficients retained, plus O(n+p) work arrays.
    // No full fitted/residual matrix is duplicated for each lambda.
    let mut coefficient_path = vec![0.0_f64; p * q * nlambda];
    let mut objective = vec![0.0_f64; nlambda];
    let mut weighted_rss = vec![0.0_f64; nlambda];
    let mut penalty_value = vec![0.0_f64; nlambda];
    let mut iterations = vec![0_i32; nlambda];
    let mut converged = vec![true; nlambda];
    let mut kkt_violation = vec![0.0_f64; nlambda];

    for r in 0..q {
        let mut beta = vec![0.0_f64; p];
        let mut residual: Vec<f64> = (0..n).map(|i| y[(i,r)] - offset[i]).collect();
        // A zero solution at the largest lambda is a valid warm initialization;
        // each subsequent lambda reuses beta AND its corresponding residuals.
        for (k, &lambda) in lambdas.iter().enumerate() {
            let mut found = false;
            let mut last_kkt = f64::INFINITY;
            for sweep in 1..=(max_iter as usize) {
                for j in 0..p {
                    let l1 = lambda * alpha * penalty[j];
                    let l2 = lambda * (1.0 - alpha) * penalty[j];
                    let denom = curvature[j] + l2;
                    if denom == 0.0 {
                        // All-zero and unpenalized: retain deterministic zero.
                        continue;
                    }
                    let mut grad = 0.0;
                    for i in 0..n {
                        grad += weights[i] * x[(i,j)] * residual[i];
                    }
                    let partial = grad / sum_w + curvature[j] * beta[j];
                    let soft = if partial > l1 { partial - l1 }
                               else if partial < -l1 { partial + l1 }
                               else { 0.0 };
                    let next = soft / denom;
                    if !next.is_finite() {
                        return Err(Error::Other("Path coordinate descent produced a non-finite coefficient.".into()));
                    }
                    let delta = next - beta[j];
                    if delta != 0.0 {
                        beta[j] = next;
                        for i in 0..n {
                            residual[i] -= x[(i,j)] * delta;
                        }
                    }
                }

                let mut kkt = 0.0_f64;
                let mut scale = 0.0_f64;
                for j in 0..p {
                    let mut grad = 0.0;
                    for i in 0..n {
                        grad += weights[i] * x[(i,j)] * residual[i];
                    }
                    grad /= sum_w;
                    let smooth = grad - lambda * (1.0 - alpha) * penalty[j] * beta[j];
                    let thresh = lambda * alpha * penalty[j];
                    let violation = if beta[j] > 0.0 { (smooth - thresh).abs() }
                        else if beta[j] < 0.0 { (smooth + thresh).abs() }
                        else { (smooth.abs() - thresh).max(0.0) };
                    kkt = kkt.max(violation);
                    scale = scale.max(grad.abs());
                }
                if !kkt.is_finite() {
                    return Err(Error::Other("Non-finite path KKT residual.".into()));
                }
                last_kkt = kkt;
                iterations[k] = iterations[k].max(sweep as i32);
                if kkt <= tol * (1.0 + scale) {
                    found = true;
                    break;
                }
            }
            converged[k] &= found;
            kkt_violation[k] = kkt_violation[k].max(last_kkt);
            for j in 0..p {
                coefficient_path[k*p*q + r*p + j] = beta[j];
                penalty_value[k] += penalty[j] *
                    (alpha * beta[j].abs() + 0.5 * (1.0 - alpha) * beta[j] * beta[j]);
            }
            for i in 0..n {
                weighted_rss[k] += weights[i] * residual[i] * residual[i];
            }
        }
    }
    for (k, &lam) in lambdas.iter().enumerate() {
        objective[k] = weighted_rss[k] / (2.0 * sum_w) + lam * penalty_value[k];
        if !objective[k].is_finite() {
            return Err(Error::Other("Non-finite path objective; rescale inputs.".into()));
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
        n_coefficients = p as i32,
        n_responses = q as i32,
        n_lambda = nlambda as i32
    ))
}
