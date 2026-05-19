// scr/core/storage.rs
use extendr_api::prelude::*;
use faer::{Mat, MatRef};

pub enum InputMatrix<'a> {
    Dense(MatRef<'a, f64>),

    SparseCscOwned {
        nrows: usize,
        ncols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f64>,
    },

    CustomPlaceholder(&'a Robj),
}

impl<'a> InputMatrix<'a> {
    pub fn try_from_robj(obj: &'a Robj) -> extendr_api::Result<Self> {
        if obj.is_matrix() {
            let nr = obj.nrows();
            let nc = obj.ncols();

            let slice = obj.as_real_slice().ok_or_else(|| {
                extendr_api::Error::Other(
                    "Expected a numeric dense matrix, but could not extract a real slice."
                        .to_string(),
                )
            })?;

            if slice.len() != nr * nc {
                return Err(extendr_api::Error::Other(format!(
                    "Dense matrix length mismatch: got length {}, but dimensions are {} x {}.",
                    slice.len(),
                    nr,
                    nc
                )));
            }

            Ok(Self::Dense(MatRef::from_column_major_slice(slice, nr, nc)))
        } else if obj.inherits("dgCMatrix") {
            let nr = obj.nrows();
            let nc = obj.ncols();

            let p_robj = obj.dollar("p")?;
            let col_ptrs: Vec<usize> = p_robj
                .as_integer_slice()
                .ok_or_else(|| {
                    extendr_api::Error::Other(
                        "Sparse dgCMatrix is missing or has corrupt 'p' slot.".to_string(),
                    )
                })?
                .iter()
                .map(|x| *x as usize)
                .collect();

            let i_robj = obj.dollar("i")?;
            let row_indices: Vec<usize> = i_robj
                .as_integer_slice()
                .ok_or_else(|| {
                    extendr_api::Error::Other(
                        "Sparse dgCMatrix is missing or has corrupt 'i' slot.".to_string(),
                    )
                })?
                .iter()
                .map(|x| *x as usize)
                .collect();

            let x_robj = obj.dollar("x")?;
            let values: Vec<f64> = x_robj
                .as_real_slice()
                .ok_or_else(|| {
                    extendr_api::Error::Other(
                        "Sparse dgCMatrix is missing or has corrupt 'x' slot.".to_string(),
                    )
                })?
                .to_vec();

            Ok(Self::SparseCscOwned {
                nrows: nr,
                ncols: nc,
                col_ptrs,
                row_indices,
                values,
            })
        } else {
            Ok(Self::CustomPlaceholder(obj))
        }
    }

    pub fn nrows(&self) -> usize {
        match self {
            Self::Dense(m) => m.nrows(),
            Self::SparseCscOwned { nrows, .. } => *nrows,
            Self::CustomPlaceholder(obj) => obj.nrows(),
        }
    }

    pub fn ncols(&self) -> usize {
        match self {
            Self::Dense(m) => m.ncols(),
            Self::SparseCscOwned { ncols, .. } => *ncols,
            Self::CustomPlaceholder(obj) => obj.ncols(),
        }
    }
}

pub fn dense_matrix_from_robj(obj: &Robj, role: &str) -> extendr_api::Result<Mat<f64>> {
    let mat = InputMatrix::try_from_robj(obj)?;

    match mat {
        InputMatrix::Dense(m) => Ok(m.to_owned()),

        InputMatrix::SparseCscOwned { .. } => Err(extendr_api::Error::Other(format!(
            "{role} was supplied as a sparse matrix. Sparse storage is preserved, but this solver does not yet implement a sparse numerical path."
        ))),

        InputMatrix::CustomPlaceholder(_) => Err(extendr_api::Error::Other(format!(
            "{role} must be a numeric dense matrix."
        ))),
    }
}

fn robj_to_f64_vec(obj: &Robj, role: &str) -> extendr_api::Result<Vec<f64>> {
    if let Some(slice) = obj.as_real_slice() {
        Ok(slice.to_vec())
    } else if let Some(slice) = obj.as_integer_slice() {
        Ok(slice.iter().map(|&x| x as f64).collect())
    } else {
        Err(extendr_api::Error::Other(format!(
            "{role} must be a numeric vector."
        )))
    }
}

pub fn numeric_vector_or_matrix_from_robj(
    obj: &Robj,
    role: &str,
    default_ncols: usize,
) -> extendr_api::Result<Mat<f64>> {
    let values = robj_to_f64_vec(obj, role)?;

    let (nr, nc) = if obj.is_matrix() {
        (obj.nrows(), obj.ncols())
    } else {
        if default_ncols == 0 || values.len() % default_ncols != 0 {
            return Err(extendr_api::Error::Other(format!(
                "{role} length {} is incompatible with default_ncols = {}.",
                values.len(),
                default_ncols
            )));
        }

        (values.len() / default_ncols, default_ncols)
    };

    if values.len() != nr * nc {
        return Err(extendr_api::Error::Other(format!(
            "{role} length mismatch: got length {}, but dimensions are {} x {}.",
            values.len(),
            nr,
            nc
        )));
    }

    Ok(MatRef::from_column_major_slice(&values, nr, nc).to_owned())
}

pub fn optional_numeric_vector_from_robj(
    obj: &Robj,
    n: usize,
    default_value: f64,
    role: &str,
) -> extendr_api::Result<Vec<f64>> {
    if obj.is_null() {
        return Ok(vec![default_value; n]);
    }

    let values = robj_to_f64_vec(obj, role)?;

    if values.len() != n {
        return Err(extendr_api::Error::Other(format!(
            "{role} has length {}, but expected length {}.",
            values.len(),
            n
        )));
    }

    Ok(values)
}

pub enum CovarianceKernel {
    GeneralPsd(Vec<MatrixStorage>),
    FactorizedZ {
        z_matrices: Vec<MatrixStorage>,
        is_block_diagonal: bool,
    },
    VarianceComponents {
        matrices: Vec<MatrixStorage>,
    },
}

pub enum MatrixStorage {
    Dense(faer::Mat<f64>),

    SparseCscOwned {
        nrows: usize,
        ncols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f64>,
    },
}