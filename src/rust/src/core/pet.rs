// src/core/pet.rs
use extendr_api::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pet {
    Ols,
    Ls,
    Wls,
    Mle,
    Reml,
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
            _ => Err(Error::Other(format!(
                "Unknown parameter estimation target: {s}. Expected OLS, LS, WLS, MLE, or REML."
            ))),
        }
    }
}