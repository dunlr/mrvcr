// src/models/linear/lm.rs

use dyn_stack::{MemBuffer, MemStack};
use extendr_api::prelude::*;
use faer::{prelude::*, Conj, Mat, MatRef, Par};

use crate::core::{
    method::ClosedFormSolver,
    storage::{dense_matrix_from_robj, optional_numeric_vector_from_robj},
    ApproximationMethod,
    Pet,
};

#[derive(Debug)]
struct LeastSquaresResult {
    beta: Mat<f64>,
    effects: Option<Mat<f64>>,
    rank: usize,
    pivoted: bool,
    method: &'static str,
    pivot_0based: Option<Vec<usize>>,
}

fn check_all_finite(m: &Mat<f64>, role: &str) -> extendr_api::Result<()> {
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            let value = m[(i, j)];

            if !value.is_finite() {
                return Err(extendr_api::Error::Other(format!(
                    "{role} contains a non-finite value at row {}, column {}.",
                    i + 1,
                    j + 1
                )));
            }
        }
    }

    Ok(())
}

fn check_all_finite_ref(m: MatRef<'_, f64>, role: &str) -> extendr_api::Result<()> {
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            let value = m[(i, j)];

            if !value.is_finite() {
                return Err(extendr_api::Error::Other(format!(
                    "{role} contains a non-finite value at row {}, column {}.",
                    i + 1,
                    j + 1
                )));
            }
        }
    }

    Ok(())
}

fn check_vector_finite(v: &[f64], role: &str) -> extendr_api::Result<()> {
    for (i, value) in v.iter().enumerate() {
        if !value.is_finite() {
            return Err(extendr_api::Error::Other(format!(
                "{role} contains a non-finite value at row {}.",
                i + 1
            )));
        }
    }

    Ok(())
}

fn check_weights(weights: &[f64]) -> extendr_api::Result<()> {
    for (i, w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(extendr_api::Error::Other(format!(
                "weights contains a missing or non-finite value at row {}.",
                i + 1
            )));
        }

        if *w < 0.0 {
            return Err(extendr_api::Error::Other(format!(
                "weights contains a negative value at row {}.",
                i + 1
            )));
        }
    }

    Ok(())
}

fn mat_col_major_vec(m: &Mat<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.nrows() * m.ncols());

    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            out.push(m[(i, j)]);
        }
    }

    out
}

fn dense_matrix_ref_from_robj<'a>(
    obj: &'a Robj,
    role: &str,
) -> extendr_api::Result<MatRef<'a, f64>> {
    use extendr_api::Error;

    if !obj.is_matrix() {
        return Err(Error::Other(format!("{role} must be a numeric matrix.")));
    }

    let nr = obj.nrows();
    let nc = obj.ncols();

    let slice = obj
        .as_real_slice()
        .ok_or_else(|| Error::Other(format!("{role} must be a double/numeric matrix.")))?;

    if slice.len() != nr * nc {
        return Err(Error::Other(format!(
            "{role} length mismatch: got {}, expected {} x {} = {}.",
            slice.len(),
            nr,
            nc,
            nr * nc
        )));
    }

    Ok(MatRef::from_column_major_slice(slice, nr, nc))
}

fn response_matrix_from_robj(obj: &Robj, role: &str) -> extendr_api::Result<Mat<f64>> {
    use extendr_api::Error;

    let slice = obj.as_real_slice().ok_or_else(|| {
        Error::Other(format!("{role} must be a double/numeric vector or matrix."))
    })?;

    if obj.is_matrix() {
        let nr = obj.nrows();
        let nc = obj.ncols();

        if slice.len() != nr * nc {
            return Err(Error::Other(format!(
                "{role} length mismatch: got {}, expected {} x {} = {}.",
                slice.len(),
                nr,
                nc,
                nr * nc
            )));
        }

        Ok(MatRef::from_column_major_slice(slice, nr, nc).to_owned())
    } else {
        Ok(MatRef::from_column_major_slice(slice, slice.len(), 1).to_owned())
    }
}

fn response_matrix_ref_from_robj<'a>(
    obj: &'a Robj,
    n_expected: usize,
    role: &str,
) -> extendr_api::Result<MatRef<'a, f64>> {
    use extendr_api::Error;

    let slice = obj.as_real_slice().ok_or_else(|| {
        Error::Other(format!("{role} must be a double/numeric vector or matrix."))
    })?;

    let (nr, nc) = if obj.is_matrix() {
        (obj.nrows(), obj.ncols())
    } else {
        (slice.len(), 1)
    };

    if nr != n_expected {
        return Err(Error::Other(format!(
            "{role} has {} rows, but expected {}.",
            nr, n_expected
        )));
    }

    if slice.len() != nr * nc {
        return Err(Error::Other(format!(
            "{role} length mismatch: got {}, expected {} x {} = {}.",
            slice.len(),
            nr,
            nc,
            nr * nc
        )));
    }

    Ok(MatRef::from_column_major_slice(slice, nr, nc))
}

fn make_weighted_problem(
    y: &Mat<f64>,
    x: &Mat<f64>,
    weights: &[f64],
    offset: &[f64],
) -> (Mat<f64>, Mat<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    let r = y.ncols();

    let mut y_work = Mat::<f64>::zeros(n, r);
    let mut x_work = Mat::<f64>::zeros(n, p);

    for i in 0..n {
        let sw = weights[i].sqrt();

        for response in 0..r {
            y_work[(i, response)] = (y[(i, response)] - offset[i]) * sw;
        }

        for j in 0..p {
            x_work[(i, j)] = x[(i, j)] * sw;
        }
    }

    (y_work, x_work)
}

fn estimate_rank_from_r_diag(
    r: MatRef<'_, f64>,
    tol: f64,
    nrows: usize,
    ncols: usize,
) -> usize {
    let diag_len = nrows.min(ncols);

    if diag_len == 0 {
        return 0;
    }

    let mut max_diag = 0.0_f64;

    for k in 0..diag_len {
        max_diag = max_diag.max(r[(k, k)].abs());
    }

    if max_diag == 0.0 || !max_diag.is_finite() {
        return 0;
    }

    let threshold = tol * max_diag;
    let mut rank = 0usize;

    for k in 0..diag_len {
        let d = r[(k, k)].abs();

        if d.is_finite() && d > threshold {
            rank += 1;
        }
    }

    rank
}

fn permutation_forward_to_usize_vec(forward: &[usize]) -> Vec<usize> {
    forward.to_vec()
}

fn coefficients_from_beta(beta: &Mat<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(beta.nrows() * beta.ncols());

    for response in 0..beta.ncols() {
        for j in 0..beta.nrows() {
            out.push(beta[(j, response)]);
        }
    }

    out
}

fn coefficients_from_beta_with_pivot_aliases(
    beta: &Mat<f64>,
    rank: usize,
    pivot_0based: Option<&[usize]>,
) -> Vec<f64> {
    let p = beta.nrows();
    let r = beta.ncols();

    let mut coefficients = coefficients_from_beta(beta);

    if rank >= p {
        return coefficients;
    }

    for response in 0..r {
        let response_offset = response * p;

        match pivot_0based {
            Some(pivot) if pivot.len() == p => {
                for k in rank..p {
                    let original_col = pivot[k];

                    if original_col < p {
                        coefficients[response_offset + original_col] = f64::NAN;
                    }
                }
            }
            _ => {
                for k in rank..p {
                    coefficients[response_offset + k] = f64::NAN;
                }
            }
        }
    }

    coefficients
}

fn pivot_1based_or_empty(pivot_0based: Option<&[usize]>) -> Vec<i32> {
    match pivot_0based {
        Some(pivot) => pivot
            .iter()
            .map(|j| (*j + 1) as i32)
            .collect::<Vec<i32>>(),
        None => Vec::<i32>::new(),
    }
}

fn compute_effects_from_qr_parts(
    q_basis: MatRef<'_, f64>,
    q_coeff: MatRef<'_, f64>,
    y: &Mat<f64>,
) -> Mat<f64> {
    let mut effects = y.clone();

    let block_size = q_coeff.nrows();

    let scratch_req =
        faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<f64>(
            q_basis.nrows(),
            block_size,
            y.ncols(),
        );

    let mut mem = MemBuffer::new(scratch_req);
    let mut stack = MemStack::new(&mut mem);

    faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
        q_basis,
        q_coeff,
        Conj::Yes,
        effects.as_mut(),
        Par::Seq,
        &mut stack,
    );

    effects
}

fn compute_effects_from_qr_parts_view(
    q_basis: MatRef<'_, f64>,
    q_coeff: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
) -> Mat<f64> {
    let y_owned = y.to_owned();

    compute_effects_from_qr_parts(q_basis, q_coeff, &y_owned)
}

fn solve_least_squares(
    x: &Mat<f64>,
    y: &Mat<f64>,
    method: ApproximationMethod,
    tol: f64,
    singular_ok: bool,
) -> extendr_api::Result<LeastSquaresResult> {
    use extendr_api::Error;

    let p = x.ncols();

    match method {
        ApproximationMethod::ClosedForm(ClosedFormSolver::QrHouseholder) => {
            if x.nrows() < x.ncols() && !singular_ok {
                return Err(Error::Other(format!(
                    "QR least squares has n < p and singular_ok = FALSE. Got n = {}, p = {}.",
                    x.nrows(),
                    x.ncols()
                )));
            }

            let decomp = x.qr();

            let effects = compute_effects_from_qr_parts(
                decomp.Q_basis(),
                decomp.Q_coeff(),
                y,
            );

            let beta = decomp.solve_lstsq(y);
            check_all_finite(&beta, "estimated coefficients")?;

            Ok(LeastSquaresResult {
                beta,
                effects: Some(effects),
                rank: p,
                pivoted: false,
                method: "householder_qr",
                pivot_0based: None,
            })
        }

        ApproximationMethod::ClosedForm(ClosedFormSolver::ColPivQr) => {
            if x.nrows() < x.ncols() && !singular_ok {
                return Err(Error::Other(format!(
                    "Column-pivoted QR least squares has n < p and singular_ok = FALSE. Got n = {}, p = {}.",
                    x.nrows(),
                    x.ncols()
                )));
            }

            let decomp = x.col_piv_qr();
            let r = decomp.R();
            let rank = estimate_rank_from_r_diag(r, tol, x.nrows(), x.ncols());

            if rank < p && !singular_ok {
                return Err(Error::Other(format!(
                    "singular fit encountered: estimated rank {} < number of columns {}.",
                    rank, p
                )));
            }

            let effects = compute_effects_from_qr_parts(
                decomp.Q_basis(),
                decomp.Q_coeff(),
                y,
            );

            let beta = decomp.solve_lstsq(y);
            check_all_finite(&beta, "estimated coefficients")?;

            let (forward, _inverse) = decomp.P().arrays();
            let pivot_0based = permutation_forward_to_usize_vec(forward);

            Ok(LeastSquaresResult {
                beta,
                effects: Some(effects),
                rank,
                pivoted: rank < p,
                method: "col_piv_qr",
                pivot_0based: Some(pivot_0based),
            })
        }

        ApproximationMethod::ClosedForm(ClosedFormSolver::Cholesky) => {
            let xtx = x.transpose() * x;
            let xty = x.transpose() * y;

            let beta = xtx
                .llt(faer::Side::Lower)
                .map_err(|_| {
                    Error::Other(
                        "LLT/Cholesky failed. X'X may be singular or not positive definite."
                            .to_string(),
                    )
                })?
                .solve(&xty);

            check_all_finite(&beta, "estimated coefficients")?;

            Ok(LeastSquaresResult {
                beta,
                effects: None,
                rank: p,
                pivoted: false,
                method: "cholesky",
                pivot_0based: None,
            })
        }

        ApproximationMethod::ClosedForm(ClosedFormSolver::Dqrls) => Err(Error::Other(
            "method = 'dqrls' / 'r_qr' is reserved for a future Rust-native R-style QR backend, but is not implemented yet.".to_string(),
        )),

        ApproximationMethod::ClosedForm(ClosedFormSolver::Svd) => Err(Error::Other(
            "SVD solver is not implemented yet for lm_fit_rust.".to_string(),
        )),

        _ => Err(Error::Other(
            "Linear model fitting requires a closed-form method: QR, householder_qr, Cholesky, or SVD."
                .to_string(),
        )),
    }
}

fn solve_least_squares_view(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    method: ApproximationMethod,
    tol: f64,
    singular_ok: bool,
) -> extendr_api::Result<LeastSquaresResult> {
    use extendr_api::Error;

    let p = x.ncols();

    match method {
        ApproximationMethod::ClosedForm(ClosedFormSolver::QrHouseholder) => {
            if x.nrows() < x.ncols() && !singular_ok {
                return Err(Error::Other(format!(
                    "QR least squares has n < p and singular_ok = FALSE. Got n = {}, p = {}.",
                    x.nrows(),
                    x.ncols()
                )));
            }

            let decomp = x.qr();

            let effects = compute_effects_from_qr_parts_view(
                decomp.Q_basis(),
                decomp.Q_coeff(),
                y,
            );

            let beta = decomp.solve_lstsq(&y);
            check_all_finite(&beta, "estimated coefficients")?;

            Ok(LeastSquaresResult {
                beta,
                effects: Some(effects),
                rank: p,
                pivoted: false,
                method: "householder_qr",
                pivot_0based: None,
            })
        }

        ApproximationMethod::ClosedForm(ClosedFormSolver::ColPivQr) => {
            if x.nrows() < x.ncols() && !singular_ok {
                return Err(Error::Other(format!(
                    "Column-pivoted QR least squares has n < p and singular_ok = FALSE. Got n = {}, p = {}.",
                    x.nrows(),
                    x.ncols()
                )));
            }

            let decomp = x.col_piv_qr();
            let r = decomp.R();
            let rank = estimate_rank_from_r_diag(r, tol, x.nrows(), x.ncols());

            if rank < p && !singular_ok {
                return Err(Error::Other(format!(
                    "singular fit encountered: estimated rank {} < number of columns {}.",
                    rank, p
                )));
            }

            let effects = compute_effects_from_qr_parts_view(
                decomp.Q_basis(),
                decomp.Q_coeff(),
                y,
            );

            let beta = decomp.solve_lstsq(&y);
            check_all_finite(&beta, "estimated coefficients")?;

            let (forward, _inverse) = decomp.P().arrays();
            let pivot_0based = permutation_forward_to_usize_vec(forward);

            Ok(LeastSquaresResult {
                beta,
                effects: Some(effects),
                rank,
                pivoted: rank < p,
                method: "col_piv_qr",
                pivot_0based: Some(pivot_0based),
            })
        }

        ApproximationMethod::ClosedForm(ClosedFormSolver::Cholesky) => {
            let xtx = x.transpose() * x;
            let xty = x.transpose() * y;

            let beta = xtx
                .llt(faer::Side::Lower)
                .map_err(|_| {
                    Error::Other(
                        "LLT/Cholesky failed. X'X may be singular or not positive definite."
                            .to_string(),
                    )
                })?
                .solve(&xty);

            check_all_finite(&beta, "estimated coefficients")?;

            Ok(LeastSquaresResult {
                beta,
                effects: None,
                rank: p,
                pivoted: false,
                method: "cholesky",
                pivot_0based: None,
            })
        }

        ApproximationMethod::ClosedForm(ClosedFormSolver::Dqrls) => Err(Error::Other(
            "method = 'dqrls' / 'r_qr' is reserved for a future Rust-native R-style QR backend, but is not implemented yet.".to_string(),
        )),

        ApproximationMethod::ClosedForm(ClosedFormSolver::Svd) => Err(Error::Other(
            "SVD solver is not implemented yet for lm_fit_rust.".to_string(),
        )),

        _ => Err(Error::Other(
            "Linear model fitting requires a closed-form method: QR, householder_qr, Cholesky, or SVD."
                .to_string(),
        )),
    }
}

fn compute_fitted_residuals_owned(
    y: &Mat<f64>,
    x: &Mat<f64>,
    beta: &Mat<f64>,
    offset: &[f64],
    weights: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    let r = y.ncols();

    let mut fitted_values = Vec::with_capacity(n * r);
    let mut residual_values = Vec::with_capacity(n * r);
    let mut rss = vec![0.0_f64; r];

    for response in 0..r {
        for i in 0..n {
            let mut fitted_i = offset[i];

            for j in 0..p {
                let b = beta[(j, response)];

                if b.is_finite() {
                    fitted_i += x[(i, j)] * b;
                }
            }

            let resid_i = y[(i, response)] - fitted_i;

            fitted_values.push(fitted_i);
            residual_values.push(resid_i);
            rss[response] += weights[i] * resid_i * resid_i;
        }
    }

    (fitted_values, residual_values, rss)
}

fn compute_fitted_residuals_view(
    y: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    beta: &Mat<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    let r = y.ncols();

    let mut fitted_values = Vec::with_capacity(n * r);
    let mut residual_values = Vec::with_capacity(n * r);
    let mut rss = vec![0.0_f64; r];

    for response in 0..r {
        for i in 0..n {
            let mut fitted_i = 0.0_f64;

            for j in 0..p {
                let b = beta[(j, response)];

                if b.is_finite() {
                    fitted_i += x[(i, j)] * b;
                }
            }

            let resid_i = y[(i, response)] - fitted_i;

            fitted_values.push(fitted_i);
            residual_values.push(resid_i);
            rss[response] += resid_i * resid_i;
        }
    }

    (fitted_values, residual_values, rss)
}

fn sigma2_from_rss(rss: &[f64], df_residual: usize, n_responses: usize) -> Vec<f64> {
    if df_residual > 0 {
        rss.iter()
            .map(|value| *value / df_residual as f64)
            .collect::<Vec<f64>>()
    } else {
        vec![f64::NAN; n_responses]
    }
}

fn fit_core(
    y_obj: Robj,
    x_obj: Robj,
    weights: Vec<f64>,
    offset_obj: Robj,
    pet_str: &str,
    method_str: &str,
    tol: f64,
    singular_ok: bool,
    calc_se: bool,
    weighted: bool,
) -> extendr_api::Result<List> {
    use extendr_api::{list, Error};

    let pet = Pet::try_from(pet_str)?;
    let method = ApproximationMethod::try_from(method_str)?;

    if weighted {
        if pet != Pet::Ols && pet != Pet::Ls && pet != Pet::Wls {
            return Err(Error::Other(
                "wlm_fit_rust supports pet = 'OLS', 'LS', or 'WLS'.".to_string(),
            ));
        }
    } else if pet != Pet::Ols && pet != Pet::Ls {
        return Err(Error::Other(
            "lm_fit_rust supports pet = 'OLS' or 'LS'.".to_string(),
        ));
    }

    let y = response_matrix_from_robj(&y_obj, "y")?;
    let x = dense_matrix_from_robj(&x_obj, "x")?;

    let n = x.nrows();
    let p = x.ncols();
    let n_responses = y.ncols();

    if n == 0 {
        return Err(Error::Other("0 non-NA cases.".to_string()));
    }

    if y.nrows() != n {
        return Err(Error::Other(format!(
            "Dimension mismatch: x has {} rows, but y has {} rows.",
            n,
            y.nrows()
        )));
    }

    if p == 0 {
        let offset = optional_numeric_vector_from_robj(&offset_obj, n, 0.0, "offset")?;

        let mut fitted_values = Vec::with_capacity(n * n_responses);
        let mut residuals = Vec::with_capacity(n * n_responses);
        let mut rss = vec![0.0_f64; n_responses];

        for response in 0..n_responses {
            for i in 0..n {
                let fitted_i = offset[i];
                let resid_i = y[(i, response)] - fitted_i;

                fitted_values.push(fitted_i);
                residuals.push(resid_i);
                rss[response] += weights[i] * resid_i * resid_i;
            }
        }

        let sigma2 = sigma2_from_rss(&rss, n, n_responses);

        return Ok(list!(
            coefficients = Vec::<f64>::new(),
            fitted_values = fitted_values,
            residuals = residuals,
            effects = Vec::<f64>::new(),
            has_effects = false,
            sigma2 = sigma2,
            rss = rss,
            rank = 0_i32,
            df_residual = n as i32,
            weights = weights,
            pivoted = false,
            pivot = Vec::<i32>::new(),
            n_responses = n_responses as i32,
            solver_method = if weighted { "wlm-empty" } else { "lm-empty" },
            converged = true,
            engine = if weighted { "faer-wlm-fit" } else { "faer-lm-fit" }
        ));
    }

    check_all_finite(&x, "x")?;
    check_all_finite(&y, "y")?;

    if weights.len() != n {
        return Err(Error::Other(format!(
            "weights has length {}, but expected {}.",
            weights.len(),
            n
        )));
    }

    check_weights(&weights)?;

    let offset = optional_numeric_vector_from_robj(&offset_obj, n, 0.0, "offset")?;
    check_vector_finite(&offset, "offset")?;

    let (y_work, x_work) = make_weighted_problem(&y, &x, &weights, &offset);
    let ls = solve_least_squares(&x_work, &y_work, method, tol, singular_ok)?;

    let beta = ls.beta;
    let effects = ls.effects;
    let rank = ls.rank;
    let pivoted = ls.pivoted;
    let solver_method = ls.method;
    let pivot_0based = ls.pivot_0based;

    let coefficients =
        coefficients_from_beta_with_pivot_aliases(&beta, rank, pivot_0based.as_deref());

    let pivot = pivot_1based_or_empty(pivot_0based.as_deref());

    let has_effects = effects.is_some();
    let effects_vec = effects
        .as_ref()
        .map(mat_col_major_vec)
        .unwrap_or_else(Vec::<f64>::new);

    let (fitted_values, residual_values, rss) =
        compute_fitted_residuals_owned(&y, &x, &beta, &offset, &weights);

    let nonzero_weight_count = weights.iter().filter(|w| **w != 0.0).count();
    let df_residual = nonzero_weight_count.saturating_sub(rank);
    let sigma2 = sigma2_from_rss(&rss, df_residual, n_responses);

    if calc_se && n_responses != 1 {
        return Err(Error::Other(
            "calc_se = TRUE is not implemented yet for multi-response linear models.".to_string(),
        ));
    }

    if calc_se {
        if rank < p {
            return Err(Error::Other(
                "Cannot compute standard errors for a rank-deficient fit yet.".to_string(),
            ));
        }

        let xtx = x_work.transpose() * &x_work;
        let identity = Mat::<f64>::identity(p, p);
        let mut vcov = xtx.qr().solve(&identity);

        for j in 0..vcov.ncols() {
            for i in 0..vcov.nrows() {
                vcov[(i, j)] *= sigma2[0];
            }
        }

        check_all_finite(&vcov, "variance-covariance matrix")?;

        Ok(list!(
            coefficients = coefficients,
            fitted_values = fitted_values,
            residuals = residual_values,
            effects = effects_vec.clone(),
            has_effects = has_effects,
            vcov = mat_col_major_vec(&vcov),
            sigma2 = sigma2,
            rss = rss,
            rank = rank as i32,
            df_residual = df_residual as i32,
            weights = weights,
            pivoted = pivoted,
            pivot = pivot.clone(),
            n_responses = n_responses as i32,
            solver_method = solver_method,
            converged = true,
            engine = if weighted { "faer-wlm-fit" } else { "faer-lm-fit" }
        ))
    } else {
        Ok(list!(
            coefficients = coefficients,
            fitted_values = fitted_values,
            residuals = residual_values,
            effects = effects_vec.clone(),
            has_effects = has_effects,
            sigma2 = sigma2,
            rss = rss,
            rank = rank as i32,
            df_residual = df_residual as i32,
            weights = weights,
            pivoted = pivoted,
            pivot = pivot.clone(),
            n_responses = n_responses as i32,
            solver_method = solver_method,
            converged = true,
            engine = if weighted { "faer-wlm-fit" } else { "faer-lm-fit" }
        ))
    }
}

fn fit_unweighted_no_offset_view(
    y_obj: Robj,
    x_obj: Robj,
    pet_str: &str,
    method_str: &str,
    tol: f64,
    singular_ok: bool,
) -> extendr_api::Result<List> {
    use extendr_api::{list, Error};

    let pet = Pet::try_from(pet_str)?;
    let method = ApproximationMethod::try_from(method_str)?;

    if pet != Pet::Ols && pet != Pet::Ls {
        return Err(Error::Other(
            "lm_fit_rust supports pet = 'OLS' or 'LS'.".to_string(),
        ));
    }

    let x = dense_matrix_ref_from_robj(&x_obj, "x")?;
    let y = response_matrix_ref_from_robj(&y_obj, x.nrows(), "y")?;

    let n = x.nrows();
    let p = x.ncols();
    let n_responses = y.ncols();

    if n == 0 {
        return Err(Error::Other("0 non-NA cases.".to_string()));
    }

    if p == 0 {
        let mut residuals = Vec::with_capacity(n * n_responses);
        let mut rss = vec![0.0_f64; n_responses];

        for response in 0..n_responses {
            for i in 0..n {
                let resid_i = y[(i, response)];
                residuals.push(resid_i);
                rss[response] += resid_i * resid_i;
            }
        }

        let fitted_values = vec![0.0; n * n_responses];
        let unit_weights = vec![1.0; n];
        let sigma2 = sigma2_from_rss(&rss, n, n_responses);

        return Ok(list!(
            coefficients = Vec::<f64>::new(),
            fitted_values = fitted_values,
            residuals = residuals,
            effects = Vec::<f64>::new(),
            has_effects = false,
            sigma2 = sigma2,
            rss = rss,
            rank = 0_i32,
            df_residual = n as i32,
            weights = unit_weights,
            pivoted = false,
            pivot = Vec::<i32>::new(),
            n_responses = n_responses as i32,
            solver_method = "lm-empty",
            converged = true,
            engine = "faer-lm-fit-view-fastpath"
        ));
    }

    check_all_finite_ref(x, "x")?;
    check_all_finite_ref(y, "y")?;

    let ls = solve_least_squares_view(x, y, method, tol, singular_ok)?;

    let beta = ls.beta;
    let effects = ls.effects;
    let rank = ls.rank;
    let pivoted = ls.pivoted;
    let solver_method = ls.method;
    let pivot_0based = ls.pivot_0based;

    let coefficients =
        coefficients_from_beta_with_pivot_aliases(&beta, rank, pivot_0based.as_deref());

    let pivot = pivot_1based_or_empty(pivot_0based.as_deref());

    let has_effects = effects.is_some();
    let effects_vec = effects
        .as_ref()
        .map(mat_col_major_vec)
        .unwrap_or_else(Vec::<f64>::new);

    let (fitted_values, residual_values, rss) = compute_fitted_residuals_view(y, x, &beta);

    let df_residual = n.saturating_sub(rank);
    let sigma2 = sigma2_from_rss(&rss, df_residual, n_responses);
    let unit_weights = vec![1.0; n];

    Ok(list!(
        coefficients = coefficients,
        fitted_values = fitted_values,
        residuals = residual_values,
        effects = effects_vec.clone(),
        has_effects = has_effects,
        sigma2 = sigma2,
        rss = rss,
        rank = rank as i32,
        df_residual = df_residual as i32,
        weights = unit_weights,
        pivoted = pivoted,
        pivot = pivot.clone(),
        n_responses = n_responses as i32,
        solver_method = solver_method,
        converged = true,
        engine = "faer-lm-fit-view-fastpath"
    ))
}

fn fit_weighted_zero_weight_core(
    y_obj: Robj,
    x_obj: Robj,
    weights: Vec<f64>,
    offset_obj: Robj,
    pet_str: &str,
    method_str: &str,
    tol: f64,
    singular_ok: bool,
    calc_se: bool,
) -> extendr_api::Result<List> {
    use extendr_api::{list, Error};

    let pet = Pet::try_from(pet_str)?;
    let method = ApproximationMethod::try_from(method_str)?;

    if pet != Pet::Ols && pet != Pet::Ls && pet != Pet::Wls {
        return Err(Error::Other(
            "wlm_fit_rust supports pet = 'OLS', 'LS', or 'WLS'.".to_string(),
        ));
    }

    let y = response_matrix_from_robj(&y_obj, "y")?;
    let x = dense_matrix_from_robj(&x_obj, "x")?;

    let n = x.nrows();
    let p = x.ncols();
    let n_responses = y.ncols();

    if y.nrows() != n {
        return Err(Error::Other(format!(
            "Dimension mismatch: x has {} rows, but y has {} rows.",
            n,
            y.nrows()
        )));
    }

    check_all_finite(&x, "x")?;
    check_all_finite(&y, "y")?;
    check_weights(&weights)?;

    let offset = optional_numeric_vector_from_robj(&offset_obj, n, 0.0, "offset")?;
    check_vector_finite(&offset, "offset")?;

    let ok: Vec<usize> = weights
        .iter()
        .enumerate()
        .filter_map(|(i, w)| if *w != 0.0 { Some(i) } else { None })
        .collect();

    let n_ok = ok.len();

    if n_ok == 0 {
        let coefficients = std::iter::repeat(f64::NAN)
            .take(p * n_responses)
            .collect::<Vec<f64>>();

        let mut fitted_values = Vec::with_capacity(n * n_responses);
        let mut residuals = Vec::with_capacity(n * n_responses);

        for response in 0..n_responses {
            for i in 0..n {
                let fitted_i = offset[i];
                let resid_i = y[(i, response)] - fitted_i;

                fitted_values.push(fitted_i);
                residuals.push(resid_i);
            }
        }

        return Ok(list!(
            coefficients = coefficients,
            fitted_values = fitted_values,
            residuals = residuals,
            effects = Vec::<f64>::new(),
            has_effects = false,
            sigma2 = vec![f64::NAN; n_responses],
            rss = vec![0.0_f64; n_responses],
            rank = 0_i32,
            df_residual = 0_i32,
            weights = weights,
            pivoted = false,
            pivot = Vec::<i32>::new(),
            n_responses = n_responses as i32,
            solver_method = "zero-weight-empty",
            converged = true,
            engine = "faer-wlm-fit-zero-weights"
        ));
    }

    let mut x_ok = Mat::<f64>::zeros(n_ok, p);
    let mut y_ok = Mat::<f64>::zeros(n_ok, n_responses);
    let mut w_ok = Vec::with_capacity(n_ok);
    let mut off_ok = Vec::with_capacity(n_ok);

    for (row_new, &row_old) in ok.iter().enumerate() {
        for response in 0..n_responses {
            y_ok[(row_new, response)] = y[(row_old, response)];
        }

        for j in 0..p {
            x_ok[(row_new, j)] = x[(row_old, j)];
        }

        w_ok.push(weights[row_old]);
        off_ok.push(offset[row_old]);
    }

    let (y_work, x_work) = make_weighted_problem(&y_ok, &x_ok, &w_ok, &off_ok);
    let ls = solve_least_squares(&x_work, &y_work, method, tol, singular_ok)?;

    let beta = ls.beta;
    let effects = ls.effects;
    let rank = ls.rank;
    let pivoted = ls.pivoted;
    let solver_method = ls.method;
    let pivot_0based = ls.pivot_0based;

    let coefficients =
        coefficients_from_beta_with_pivot_aliases(&beta, rank, pivot_0based.as_deref());

    let pivot = pivot_1based_or_empty(pivot_0based.as_deref());

    let has_effects = effects.is_some();
    let effects_vec = effects
        .as_ref()
        .map(mat_col_major_vec)
        .unwrap_or_else(Vec::<f64>::new);

    let (fitted_values, residual_values, rss) =
        compute_fitted_residuals_owned(&y, &x, &beta, &offset, &weights);

    let df_residual = n_ok.saturating_sub(rank);
    let sigma2 = sigma2_from_rss(&rss, df_residual, n_responses);

    if calc_se && n_responses != 1 {
        return Err(Error::Other(
            "calc_se = TRUE is not implemented yet for multi-response linear models.".to_string(),
        ));
    }

    if calc_se {
        if rank < p {
            return Err(Error::Other(
                "Cannot compute standard errors for a rank-deficient fit yet.".to_string(),
            ));
        }

        let xtx = x_work.transpose() * &x_work;
        let identity = Mat::<f64>::identity(p, p);
        let mut vcov = xtx.qr().solve(&identity);

        for j in 0..vcov.ncols() {
            for i in 0..vcov.nrows() {
                vcov[(i, j)] *= sigma2[0];
            }
        }

        check_all_finite(&vcov, "variance-covariance matrix")?;

        Ok(list!(
            coefficients = coefficients,
            fitted_values = fitted_values,
            residuals = residual_values,
            effects = effects_vec.clone(),
            has_effects = has_effects,
            vcov = mat_col_major_vec(&vcov),
            sigma2 = sigma2,
            rss = rss,
            rank = rank as i32,
            df_residual = df_residual as i32,
            weights = weights,
            pivoted = pivoted,
            pivot = pivot.clone(),
            n_responses = n_responses as i32,
            solver_method = solver_method,
            converged = true,
            engine = "faer-wlm-fit-zero-weights"
        ))
    } else {
        Ok(list!(
            coefficients = coefficients,
            fitted_values = fitted_values,
            residuals = residual_values,
            effects = effects_vec.clone(),
            has_effects = has_effects,
            sigma2 = sigma2,
            rss = rss,
            rank = rank as i32,
            df_residual = df_residual as i32,
            weights = weights,
            pivoted = pivoted,
            pivot = pivot.clone(),
            n_responses = n_responses as i32,
            solver_method = solver_method,
            converged = true,
            engine = "faer-wlm-fit-zero-weights"
        ))
    }
}

pub fn ffi_fit(
    y_obj: Robj,
    x_obj: Robj,
    offset_obj: Robj,
    pet_str: &str,
    method_str: &str,
    tol: f64,
    singular_ok: bool,
    calc_se: bool,
) -> extendr_api::Result<List> {
    if !tol.is_finite() || tol <= 0.0 {
        return Err(extendr_api::Error::Other(
            "tol must be positive and finite.".to_string(),
        ));
    }

    if offset_obj.is_null() && !calc_se {
        return fit_unweighted_no_offset_view(
            y_obj,
            x_obj,
            pet_str,
            method_str,
            tol,
            singular_ok,
        );
    }

    let n = if x_obj.is_matrix() {
        x_obj.nrows()
    } else {
        y_obj.len()
    };

    let weights = vec![1.0; n];

    fit_core(
        y_obj,
        x_obj,
        weights,
        offset_obj,
        pet_str,
        method_str,
        tol,
        singular_ok,
        calc_se,
        false,
    )
}

pub fn ffi_wfit(
    y_obj: Robj,
    x_obj: Robj,
    weights_obj: Robj,
    offset_obj: Robj,
    pet_str: &str,
    method_str: &str,
    tol: f64,
    singular_ok: bool,
    calc_se: bool,
) -> extendr_api::Result<List> {
    if !tol.is_finite() || tol <= 0.0 {
        return Err(extendr_api::Error::Other(
            "tol must be positive and finite.".to_string(),
        ));
    }

    let n = if x_obj.is_matrix() {
        x_obj.nrows()
    } else {
        y_obj.len()
    };

    let weights = optional_numeric_vector_from_robj(&weights_obj, n, 1.0, "weights")?;
    check_weights(&weights)?;

    if weights.iter().any(|w| *w == 0.0) {
        return fit_weighted_zero_weight_core(
            y_obj,
            x_obj,
            weights,
            offset_obj,
            pet_str,
            method_str,
            tol,
            singular_ok,
            calc_se,
        );
    }

    fit_core(
        y_obj,
        x_obj,
        weights,
        offset_obj,
        pet_str,
        method_str,
        tol,
        singular_ok,
        calc_se,
        true,
    )
}