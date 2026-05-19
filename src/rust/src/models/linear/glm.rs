// src/models/linear/glm.rs
use extendr_api::prelude::*;
use faer::{Mat, prelude::*};

use crate::core::{
    method::NumericalOptimizer,
    storage::{
        dense_matrix_from_robj,
        numeric_vector_or_matrix_from_robj,
        optional_numeric_vector_from_robj,
    },
    ApproximationMethod,
    Pet,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Link {
    Identity,
    Log,
    Logit,
    Inverse,
}

impl TryFrom<&str> for Link {
    type Error = extendr_api::Error;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let key = s.trim().to_ascii_lowercase();

        match key.as_str() {
            "identity" => Ok(Self::Identity),
            "log" => Ok(Self::Log),
            "logit" => Ok(Self::Logit),
            "inverse" => Ok(Self::Inverse),
            _ => Err(extendr_api::Error::Other(format!(
                "Unknown link: {s}. Expected identity, log, logit, or inverse."
            ))),
        }
    }
}

impl Link {
    fn inverse(self, eta: f64) -> f64 {
        match self {
            Self::Identity => eta,
            Self::Log => eta.exp(),
            Self::Logit => {
                if eta >= 0.0 {
                    1.0 / (1.0 + (-eta).exp())
                } else {
                    let e = eta.exp();
                    e / (1.0 + e)
                }
            }
            Self::Inverse => 1.0 / eta,
        }
    }

    fn derivative(self, mu: f64) -> f64 {
        let eps = 1e-12;

        match self {
            Self::Identity => 1.0,
            Self::Log => 1.0 / mu.max(eps),
            Self::Logit => {
                let m = mu.clamp(eps, 1.0 - eps);
                1.0 / (m * (1.0 - m))
            }
            Self::Inverse => -1.0 / (mu * mu),
        }
    }

    fn clamp_mu(self, mu: f64) -> f64 {
        let eps = 1e-12;

        match self {
            Self::Logit => mu.clamp(eps, 1.0 - eps),
            Self::Log | Self::Inverse => mu.max(eps),
            Self::Identity => mu,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Family {
    Gaussian,
    Poisson,
    Binomial,
    NegativeBinomial { theta: f64 },
    Tweedie { power: f64 },
}

impl Family {
    fn parse(s: &str, param: f64) -> extendr_api::Result<Self> {
        let key = s.trim().to_ascii_lowercase();

        match key.as_str() {
            "gaussian" | "normal" => Ok(Self::Gaussian),
            "poisson" => Ok(Self::Poisson),
            "binomial" => Ok(Self::Binomial),

            "nb" | "negative_binomial" | "negativebinomial" => {
                if param <= 0.0 || !param.is_finite() {
                    Err(extendr_api::Error::Other(
                        "Negative binomial requires param/theta > 0.".to_string(),
                    ))
                } else {
                    Ok(Self::NegativeBinomial { theta: param })
                }
            }

            "tweedie" => {
                if !param.is_finite() {
                    Err(extendr_api::Error::Other(
                        "Tweedie requires a finite power parameter.".to_string(),
                    ))
                } else {
                    Ok(Self::Tweedie { power: param })
                }
            }

            _ => Err(extendr_api::Error::Other(format!(
                "Unknown family: {s}. Expected gaussian, poisson, binomial, nb, or tweedie."
            ))),
        }
    }

    fn variance(self, mu: f64) -> f64 {
        match self {
            Self::Gaussian => 1.0,
            Self::Poisson => mu,
            Self::Binomial => mu * (1.0 - mu),
            Self::NegativeBinomial { theta } => mu + (mu * mu) / theta,
            Self::Tweedie { power } => mu.powf(power),
        }
    }
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

fn check_weights(weights: &[f64]) -> extendr_api::Result<()> {
    for (i, w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(extendr_api::Error::Other(format!(
                "weights contains a non-finite value at row {}.",
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

fn solve_weighted_least_squares_qr(
    x: &Mat<f64>,
    z: &Mat<f64>,
    weights: &[f64],
) -> extendr_api::Result<Mat<f64>> {
    let n = x.nrows();
    let p = x.ncols();

    let mut x_work = Mat::<f64>::zeros(n, p);
    let mut z_work = Mat::<f64>::zeros(n, 1);

    for i in 0..n {
        let sw = weights[i].sqrt();

        z_work[(i, 0)] = z[(i, 0)] * sw;

        for j in 0..p {
            x_work[(i, j)] = x[(i, j)] * sw;
        }
    }

    let beta = x_work.qr().solve_lstsq(&z_work);
    check_all_finite(&beta, "estimated coefficients")?;

    Ok(beta)
}

fn compute_mu_eta_weights(
    y: &Mat<f64>,
    x: &Mat<f64>,
    beta: &Mat<f64>,
    offset: &[f64],
    prior_weights: &[f64],
    family: Family,
    link: Link,
) -> extendr_api::Result<(Mat<f64>, Mat<f64>, Vec<f64>)> {
    use extendr_api::Error;

    let n = x.nrows();

    let mut eta = x * beta;
    let mut mu = Mat::<f64>::zeros(n, 1);
    let mut weights = vec![0.0; n];

    for i in 0..n {
        eta[(i, 0)] += offset[i];

        let mu_i = link.clamp_mu(link.inverse(eta[(i, 0)]));
        let var_i = family.variance(mu_i);
        let g_prime = link.derivative(mu_i);

        if !var_i.is_finite() || var_i <= 0.0 || !g_prime.is_finite() || g_prime == 0.0 {
            return Err(Error::Other(format!(
                "Invalid IRLS weight at row {}: y = {}, mu = {}, variance = {}, g_prime = {}.",
                i + 1,
                y[(i, 0)],
                mu_i,
                var_i,
                g_prime
            )));
        }

        mu[(i, 0)] = mu_i;
        weights[i] = prior_weights[i] / (var_i * g_prime * g_prime);
    }

    Ok((eta, mu, weights))
}

pub fn ffi_fit_standard(
    y_obj: Robj,
    x_obj: Robj,
    weights_obj: Robj,
    offset_obj: Robj,
    family_str: &str,
    link_str: &str,
    pet_str: &str,
    method_str: &str,
    param: f64,
    max_iter: i32,
    tol: f64,
    _singular_ok: bool,
    _calc_se: bool,
) -> extendr_api::Result<List> {
    use extendr_api::{list, Error};

    let pet = Pet::try_from(pet_str)?;
    let method = ApproximationMethod::try_from(method_str)?;

    if pet != Pet::Mle {
        return Err(Error::Other(
            "glm_fit_rust currently supports only pet = 'MLE'.".to_string(),
        ));
    }

    match method {
        ApproximationMethod::Optimization(NumericalOptimizer::FisherScoring) => {}
        _ => {
            return Err(Error::Other(
                "glm_fit_rust currently supports only method = 'FisherScoring'.".to_string(),
            ));
        }
    }

    let family = Family::parse(family_str, param)?;
    let link = Link::try_from(link_str)?;

    let y = numeric_vector_or_matrix_from_robj(&y_obj, "y", 1)?;
    let x = dense_matrix_from_robj(&x_obj, "x")?;

    let n = x.nrows();
    let p = x.ncols();

    if y.nrows() != n || y.ncols() != 1 {
        return Err(Error::Other(format!(
            "Dimension mismatch: x has {} rows, but y has dimensions {} x {}.",
            n,
            y.nrows(),
            y.ncols()
        )));
    }

    check_all_finite(&x, "x")?;
    check_all_finite(&y, "y")?;

    let prior_weights = optional_numeric_vector_from_robj(&weights_obj, n, 1.0, "weights")?;
    let offset = optional_numeric_vector_from_robj(&offset_obj, n, 0.0, "offset")?;

    check_weights(&prior_weights)?;

    if max_iter <= 0 {
        return Err(Error::Other("max_iter must be positive.".to_string()));
    }

    if tol <= 0.0 || !tol.is_finite() {
        return Err(Error::Other("tol must be positive and finite.".to_string()));
    }

    let mut beta = Mat::<f64>::zeros(p, 1);
    let mut converged = false;
    let mut iterations = 0usize;

    for iter in 0..(max_iter as usize) {
        iterations = iter + 1;

        let (eta, mu, working_weights) = compute_mu_eta_weights(
            &y,
            &x,
            &beta,
            &offset,
            &prior_weights,
            family,
            link,
        )?;

        let mut z_for_beta = Mat::<f64>::zeros(n, 1);

        for i in 0..n {
            let mu_i = mu[(i, 0)];
            let g_prime = link.derivative(mu_i);
            let z_total = eta[(i, 0)] + (y[(i, 0)] - mu_i) * g_prime;

            // Offset is fixed, so the working response for beta removes it.
            z_for_beta[(i, 0)] = z_total - offset[i];
        }

        let next_beta = solve_weighted_least_squares_qr(
            &x,
            &z_for_beta,
            &working_weights,
        )?;

        let step = (&next_beta - &beta).norm_max();

        beta = next_beta;

        if step < tol {
            converged = true;
            break;
        }
    }

    let (final_eta, final_mu, final_working_weights) = compute_mu_eta_weights(
        &y,
        &x,
        &beta,
        &offset,
        &prior_weights,
        family,
        link,
    )?;

    let coefficients = beta.col(0).iter().copied().collect::<Vec<f64>>();
    let fitted_values = final_mu.col(0).iter().copied().collect::<Vec<f64>>();
    let linear_predictors = final_eta.col(0).iter().copied().collect::<Vec<f64>>();

    let mut residuals = Vec::with_capacity(n);
    let mut deviance_like = 0.0;

    for i in 0..n {
        let r = y[(i, 0)] - final_mu[(i, 0)];
        residuals.push(r);
        deviance_like += prior_weights[i] * r * r;
    }

    Ok(list!(
        coefficients = coefficients,
        fitted_values = fitted_values,
        linear_predictors = linear_predictors,
        residuals = residuals,
        prior_weights = prior_weights,
        weights = final_working_weights,
        deviance = deviance_like,
        iterations = iterations as i32,
        converged = converged,
        engine = "faer-glm-irls-fisher-scoring-qr"
    ))
}