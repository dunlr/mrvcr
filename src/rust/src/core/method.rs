// src/core/method.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosedFormSolver {
    QrHouseholder,
    ColPivQr,
    Dqrls,
    Svd,
    Cholesky,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericalOptimizer {
    FisherScoring,
    NewtonRaphson,
    MmAlgorithm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApproximationMethod {
    ClosedForm(ClosedFormSolver),
    Optimization(NumericalOptimizer),
}

impl TryFrom<&str> for ApproximationMethod {
    type Error = extendr_api::Error;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let key = s.trim().to_ascii_lowercase().replace('-', "_");

        match key.as_str() {
            "qr" | "col_piv_qr" | "column_pivoted_qr" | "pivoted_qr" => {
                Ok(Self::ClosedForm(ClosedFormSolver::ColPivQr))
            }

            "householder_qr" | "unpivoted_qr" | "plain_qr" => {
                Ok(Self::ClosedForm(ClosedFormSolver::QrHouseholder))
            }

            "dqrls" | "r_qr" | "r_style_qr" | "stats_qr" => {
                Ok(Self::ClosedForm(ClosedFormSolver::Dqrls))
            }

            "svd" => Ok(Self::ClosedForm(ClosedFormSolver::Svd)),

            "cholesky" | "chol" | "llt" => {
                Ok(Self::ClosedForm(ClosedFormSolver::Cholesky))
            }

            "fs" | "fisher" | "fisher_scoring" | "fisherscoring" => {
                Ok(Self::Optimization(NumericalOptimizer::FisherScoring))
            }

            "newton" | "newton_raphson" | "newtonraphson" => {
                Ok(Self::Optimization(NumericalOptimizer::NewtonRaphson))
            }

            "mm" | "mm_algorithm" | "mmalgorithm" => {
                Ok(Self::Optimization(NumericalOptimizer::MmAlgorithm))
            }

            _ => Err(extendr_api::Error::Other(format!(
                "Unknown approximation method: {s}. Supported methods include \
                 'qr', 'householder_qr', 'col_piv_qr', 'cholesky', 'svd', \
                 'dqrls', 'r_qr', 'FisherScoring', and 'MM'."
            ))),
        }
    }
}