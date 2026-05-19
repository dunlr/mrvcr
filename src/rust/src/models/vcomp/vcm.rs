// src/models/vcomp/vcm.rs
use extendr_api::prelude::*;
use faer::prelude::*;

use crate::core::{
    method::NumericalOptimizer,
    storage::{dense_matrix_from_robj, numeric_vector_or_matrix_from_robj},
    ApproximationMethod,
    Pet,
};

pub fn ffi_fit_general(
    _y_obj: Robj,
    _x_obj: Robj,
    _v_obj: Robj,
    _pet_str: &str,
    _method_str: &str,
) -> extendr_api::Result<List> {
    use extendr_api::list;

    Ok(list!(
        status = "General dense VCM track registered, but full estimator is not implemented yet."
    ))
}

pub fn ffi_fit_zzt(
    y_obj: Robj,
    x_obj: Robj,
    z_obj: Robj,
    pet_str: &str,
    method_str: &str,
) -> extendr_api::Result<List> {
    use extendr_api::{list, Error};

    let pet = Pet::try_from(pet_str)?;
    let method = ApproximationMethod::try_from(method_str)?;

    if pet != Pet::Reml && pet != Pet::Mle {
        return Err(Error::Other(
            "VCM estimation requires pet = 'MLE' or pet = 'REML'.".to_string(),
        ));
    }

    let y = numeric_vector_or_matrix_from_robj(&y_obj, "y", 1)?;
    let x = dense_matrix_from_robj(&x_obj, "x")?;
    let z = dense_matrix_from_robj(&z_obj, "z")?;

    if y.ncols() != 1 {
        return Err(Error::Other(
            "vcm_zzt currently supports only a univariate response.".to_string(),
        ));
    }

    if x.nrows() != y.nrows() || z.nrows() != y.nrows() {
        return Err(Error::Other(format!(
            "Dimension mismatch: y has {} rows, X has {} rows, Z has {} rows.",
            y.nrows(),
            x.nrows(),
            z.nrows()
        )));
    }

    match method {
        ApproximationMethod::Optimization(NumericalOptimizer::MmAlgorithm) => {
            let ztz = z.transpose() * &z;

            let cholesky = ztz
                .ldlt(faer::Side::Lower)
                .map_err(|_| {
                    Error::Other(
                        "Z'Z factorization failed. The random-effect design may be rank-deficient."
                            .to_string(),
                    )
                })?;

            let l_view = cholesky.L();

            // Placeholder parameters.
            // These are NOT a valid MM estimator yet.
            let mut sigma_e2 = 1.0;
            let mut sigma_g2 = 0.5;

            for _iter in 0..20 {
                let scale = sigma_e2 / sigma_g2;
                let mut inner_core = ztz.to_owned();

                for i in 0..z.ncols() {
                    inner_core[(i, i)] += scale;
                }

                let inner_ldlt = inner_core
                    .ldlt(faer::Side::Lower)
                    .map_err(|_| {
                        Error::Other("Woodbury inner-core factorization failed.".to_string())
                    })?;

                let zty = z.transpose() * &y;
                let solved_inner = inner_ldlt.solve(&zty);
                let z_solved = &z * &solved_inner;

                let mut _sigma_inv_y = y.to_owned();
                for i in 0..y.nrows() {
                    _sigma_inv_y[(i, 0)] = (y[(i, 0)] - z_solved[(i, 0)]) / sigma_e2;
                }

                // TODO: replace with actual ML/REML MM updates.
                sigma_e2 *= 0.95;
                sigma_g2 *= 1.02;
            }

            let beta = x.qr().solve(&y);
            let coefficients = beta.col(0).iter().copied().collect::<Vec<f64>>();

            Ok(list!(
                coefficients = coefficients,
                variance_components = vec![sigma_e2, sigma_g2],
                engine = "prototype-zzt-woodbury-mm-not-final-estimator",
                random_structure_dimension = l_view.nrows() as i32,
                converged = false,
                message = "VCM ZZT path is wired, but variance updates are placeholders."
            ))
        }

        _ => Err(Error::Other(
            "vcm_zzt currently supports only method = 'MM'.".to_string(),
        )),
    }
}