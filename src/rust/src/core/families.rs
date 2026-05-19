// src/core/families.rs
use extendr_api::prelude::Function;

// --- LINK LAYER ---
pub trait LinkFunction: Send + Sync {
    fn link(&self, mu: f64) -> f64;
    fn link_inverse(&self, eta: f64) -> f64;
    fn derivative(&self, mu: f64) -> f64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StandardLink {
    Identity,
    Log,
    Logit,
    Inverse,
}

impl LinkFunction for StandardLink {
    #[inline]
    fn link(&self, mu: f64) -> f64 {
        match self {
            Self::Identity => mu,
            Self::Log => mu.ln(),
            Self::Logit => mu.ln() - (1.0 - mu).ln(),
            Self::Inverse => 1.0 / mu,
        }
    }

    #[inline]
    fn link_inverse(&self, eta: f64) -> f64 {
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

    #[inline]
    fn derivative(&self, mu: f64) -> f64 {
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
}

// --- CUSTOM LINK LAYER ---
pub struct CustomRLink {
    pub link_fn: Function,
    pub inv_fn: Function,
    pub deriv_fn: Function,
}

// Bypassing R's strict single-threaded pointer limitations safely
unsafe impl Send for CustomRLink {}
unsafe impl Sync for CustomRLink {}

impl LinkFunction for CustomRLink {
    fn link(&self, mu: f64) -> f64 {
        self.link_fn.call(extendr_api::pairlist!(mu)).unwrap().as_real().unwrap()
    }
    fn link_inverse(&self, eta: f64) -> f64 {
        self.inv_fn.call(extendr_api::pairlist!(eta)).unwrap().as_real().unwrap()
    }
    fn derivative(&self, mu: f64) -> f64 {
        self.deriv_fn.call(extendr_api::pairlist!(mu)).unwrap().as_real().unwrap()
    }
}

// --- DISTRIBUTION FAMILY LAYER ---
pub trait DistributionFamily: Send + Sync {
    fn variance(&self, mu: f64) -> f64;
}

pub struct Gaussian;
impl DistributionFamily for Gaussian {
    fn variance(&self, _mu: f64) -> f64 {
        1.0
    }
}

pub struct Poisson;
impl DistributionFamily for Poisson {
    fn variance(&self, mu: f64) -> f64 {
        mu
    }
}

pub struct Binomial;
impl DistributionFamily for Binomial {
    fn variance(&self, mu: f64) -> f64 {
        mu * (1.0 - mu)
    }
}

pub struct NegativeBinomial {
    pub theta: f64,
}
impl DistributionFamily for NegativeBinomial {
    fn variance(&self, mu: f64) -> f64 {
        mu + (mu * mu) / self.theta
    }
}

pub struct Tweedie {
    pub power: f64,
}
impl DistributionFamily for Tweedie {
    fn variance(&self, mu: f64) -> f64 {
        mu.powf(self.power)
    }
}

pub struct ModelSpecification<F: DistributionFamily, L: LinkFunction> {
    pub family: F,
    pub link: L,
}

impl<F: DistributionFamily, L: LinkFunction> ModelSpecification<F, L> {
    pub fn new(family: F, link: L) -> Self {
        Self { family, link }
    }
}
