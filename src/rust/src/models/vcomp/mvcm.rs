// src/models/vcomp/mvcm.rs
use extendr_api::prelude::*;
use faer::prelude::*;

use crate::core::storage::{dense_matrix_from_robj, numeric_vector_or_matrix_from_robj};

pub fn ffi_fit(
    y_obj: Robj,
    x_obj: Robj,
    _pet_str: &str,
) -> extendr_api::Result<List> {
    use extendr_api::{list, Error};

    let y = numeric_vector_or_matrix_from_robj(&y_obj, "Y", 1)?;
    let x = dense_matrix_from_robj(&x_obj, "X")?;

    if x.nrows() != y.nrows() {
        return Err(Error::Other(format!(
            "Dimension mismatch: X has {} rows, but Y has {} rows.",
            x.nrows(),
            y.nrows()
        )));
    }

    let xtx = x.transpose() * &x;
    let xty = x.transpose() * &y;
    let beta_matrix = xtx.qr().solve(&xty);

    let mut coefficients_flat = Vec::with_capacity(beta_matrix.nrows() * beta_matrix.ncols());
    for col in 0..beta_matrix.ncols() {
        for row in 0..beta_matrix.nrows() {
            coefficients_flat.push(beta_matrix[(row, col)]);
        }
    }

    Ok(list!(
        coefficients_matrix_flat = coefficients_flat,
        response_count = y.ncols() as i32,
        coefficient_count_per_resp = x.ncols() as i32,
        engine = "faer-multivariate-ols-baseline"
    ))
}