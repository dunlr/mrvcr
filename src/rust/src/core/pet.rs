// Parameter estimation targets (PETs). Hyperparameters belong to the PET,
// not to the numerical method used to compute its minimizer.
use extendr_api::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pet {
    Ols,
    Ls,
    Wls,
    Mle,
    Reml,
    Ridge { lambda: f64 },
    Lasso { lambda: f64 },
    ElasticNet { lambda: f64, alpha: f64 },
}

impl Pet {
    /// `alpha` is NaN only when the R caller omitted this optional input.
    /// Never infer an elastic-net mixing parameter: the user must specify it.
    pub fn from_regularized_str(name: &str, lambda: f64, alpha: f64) -> Result<Self, Error> {
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(Error::Other("lambda must be finite and nonnegative.".to_string()));
        }
        match name.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "ridge" => {
                if alpha.is_nan() || alpha == 0.0 {
                    Ok(Self::Ridge { lambda })
                } else {
                    Err(Error::Other("pet = 'Ridge' requires alpha = 0 (or omitted).".to_string()))
                }
            }
            "lasso" => {
                if alpha.is_nan() || alpha == 1.0 {
                    Ok(Self::Lasso { lambda })
                } else {
                    Err(Error::Other("pet = 'Lasso' requires alpha = 1 (or omitted).".to_string()))
                }
            }
            "elastic_net" | "elasticnet" | "enet" => {
                if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
                    Err(Error::Other("pet = 'ElasticNet' requires a finite alpha in [0, 1].".to_string()))
                } else {
                    Ok(Self::ElasticNet { lambda, alpha })
                }
            }
            other => Err(Error::Other(format!(
                "Unsupported penalized PET '{other}'. Expected Ridge, Lasso, or ElasticNet."
            ))),
        }
    }
}

impl TryFrom<&str> for Pet {
    type Error = Error;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let key = s.trim().to_ascii_uppercase();
        match key.as_str() {
            "OLS" => Ok(Self::Ols),
            "LS" => Ok(Self::Ls),
            "WLS" => Ok(Self::Wls),
            "MLE" | "ML" => Ok(Self::Mle),
            "REML" => Ok(Self::Reml),
            "RIDGE" | "LASSO" | "ELASTICNET" | "ELASTIC_NET" => Err(Error::Other(
                "Penalized PETs require statistical hyperparameters; use a penalized fitting interface (e.g. lmnet).".to_string()
            )),
            _ => Err(Error::Other(format!(
                "Unknown parameter estimation target: {s}. Expected OLS, LS, WLS, MLE, or REML."
            ))),
        }
    }
}
