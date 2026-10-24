//! Gaussian penalized linear regression (initial PET: Ridge).
//!
//! Criterion, with S = sum_i w_i and q responses:
//!   (1 / (2 S)) sum_{i,r} w_i (y_ir - offset_i - x_i' beta_r)^2
//!       + (lambda / 2) sum_{j,r} penalty_factor_j * beta_jr^2 .
//!
//! A nonpenalized intercept corresponds to penalty_factor_j = 0.
//! No automatic standardization is applied.  The coefficient order is the
//! original (unpivoted) order and multiple responses share the factorization.

use extendr_api::prelude::*;
use faer::{Mat, MatRef, Side};
use faer::prelude::{Solve, SolveLstsq};
use crate::core::{
    method::ClosedFormSolver,
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

/// No copies of X beyond the single weighted/augmented working matrix.
/// The input R design matrix is borrowed through MatRef for the call duration.
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
) -> extendr_api::Result<List> {
    use extendr_api::{list, Error};

    let pet = Pet::from_regularized_str(pet_str, lambda, alpha)?;
    let lambda = match pet {
        Pet::Ridge { lambda } => lambda,
        Pet::Lasso { .. } => return Err(Error::Other(
            "PET 'Lasso' is registered but not implemented yet; coordinate descent is planned."
                .to_string()
        )),
        Pet::ElasticNet { .. } => return Err(Error::Other(
            "PET 'ElasticNet' is registered but not implemented yet; coordinate descent is planned."
                .to_string()
        )),
        _ => return Err(Error::Other("This endpoint currently supports Ridge only.".to_string())),
    };

    let method = ApproximationMethod::try_from(method_str)?;
    let solver = match method {
        ApproximationMethod::ClosedForm(ClosedFormSolver::Cholesky) => "cholesky",
        ApproximationMethod::ClosedForm(ClosedFormSolver::ColPivQr) => "qr",
        ApproximationMethod::ClosedForm(ClosedFormSolver::QrHouseholder) => "householder_qr",
        _ => return Err(Error::Other(
            "Ridge supports method = 'cholesky', 'qr', or 'householder_qr'.".to_string()
        )),
    };

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

    // Scale by S = sum(w), rather than n, to give weighted fits an explicit
    // penalty convention invariant under multiplying every weight by a constant.
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
        // Pivoting is helpful when some coefficients are deliberately unpenalized.
        x_work.col_piv_qr().solve_lstsq(&y_work)
    } else {
        x_work.qr().solve_lstsq(&y_work)
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
            penalty_value += penalty[j] * beta[(j,r)] * beta[(j,r)];
        }
    }
    let objective = weighted_rss / (2.0 * sum_w) + 0.5 * lambda * penalty_value;
    if !objective.is_finite() {
        return Err(Error::Other("Ridge fit resulted in a non-finite objective.".to_string()));
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
        pet = "Ridge",
        converged = true,
        n_responses = q as i32,
        n_coefficients = p as i32
    ))
}
