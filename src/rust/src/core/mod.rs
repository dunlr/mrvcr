// src/core/mod.rs
use faer::Mat;

pub mod families;
pub mod method;
pub mod pet;
pub mod storage;

pub use method::ApproximationMethod;
pub use pet::Pet;
pub use storage::InputMatrix;

/// Base trait for any parametric statistical model
pub trait StatisticalModel {
    /// Associated storage type (e.g., Dense or Sparse matrix layouts)
    type Storage;

    /// Verifies if the parameters specify the dimensions of the unknown parameter space
    fn parameter_dimension(&self) -> usize;
}

/// Marker trait for models that possess an explicit likelihood function
pub trait HasLikelihood: StatisticalModel {
    /// Computes the log-likelihood for a given vector of unknown parameters
    fn log_likelihood(&self, parameters: &Mat<f64>) -> f64;

    /// Computes analytical gradients using automatic differentiation or closed-form derivations
    fn score_function(&self, parameters: &Mat<f64>) -> Mat<f64>;
}

/// Functional execution trait for fitting
pub trait Estimatable<M: StatisticalModel>: StatisticalModel {
    fn fit(
        &self,
        pet: Pet,
        method: ApproximationMethod,
    ) -> Result<crate::models::FitResult, String>;
}
