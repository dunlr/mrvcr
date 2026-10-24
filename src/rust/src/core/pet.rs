// Parameter estimation targets (PETs).
//
// The model defines the parameter space and likelihood/loss. A PET chooses
// the estimation criterion; numerical solvers are selected independently.
// Statistical tuning parameters live in variants, rather than overloading
// numerical tolerances or solver settings.
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
    /// Parse a regularized estimation target, with statistical hyperparameters.
    /// `alpha` is the L1 fraction for elastic net, not a numerical tolerance.
    /// The zero-alpha ridge case has its own dedicated PET variant.
    pub fn from_regularized_str(name: &str, lambda: f64, alpha: f64) -> Result<Self, Error> {
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(Error::Other("lambda must be finite and nonnegative.".to_string()));
        }
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(Error::Other("alpha must be finite and between 0 and 1.".to_string()));
        }
        match name.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "ridge" => {
                if alpha != 0.0 {
                    Err(Error::Other("pet = 'Ridge' requires alpha = 0.".to_string()))
                } else {
                    Ok(Self::Ridge { lambda })
                }
            }
            "lasso" => Ok(Self::Lasso { lambda }),
            "elastic_net" | "elasticnet" | "enet" => {
                Ok(Self::ElasticNet { lambda, alpha })
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
