// src/models/mod.rs
pub mod linear;
pub mod mixed;
pub mod time_series;
pub mod vcomp;

#[derive(Debug, Clone)]
pub struct FitResult {
    pub coefficients: Vec<f64>,
    pub fitted_values: Option<Vec<f64>>,
    pub residuals: Option<Vec<f64>>,
    pub standard_errors: Option<Vec<f64>>,
    pub vcov: Option<Vec<f64>>,
    pub log_likelihood: Option<f64>,
    pub deviance: Option<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub message: String,
}