# Tests for native-only GLM family/link dispatch.
# These tests require the rebuilt Rust DLL; don't test with devtools::load_all()
# against an older installed binary.

.glm_compare <- function(formula, data, family, tolerance = 3e-4, ...) {
  fit <- mrvcr::glm(formula, data = data, family = family, ...)
  ref <- stats::glm(formula, data = data, family = family, ...)

  expect_s3_class(fit, "mrvcr_glm")
  expect_true(isTRUE(fit$converged))
  expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = tolerance)
  expect_equal(unname(fitted(fit)), unname(fitted(ref)), tolerance = tolerance)
  expect_equal(fit$deviance, ref$deviance, tolerance = tolerance)
  fit
}

test_that("Gaussian, Poisson, and Bernoulli logit fit natively", {
  set.seed(104)
  d <- data.frame(x = rnorm(400))
  d$yg <- 1 + 0.8 * d$x + rnorm(400)
  d$yp <- rpois(400, exp(0.2 + 0.3 * d$x))
  d$yb <- rbinom(400, 1, plogis(-0.1 + 0.6 * d$x))

  .glm_compare(yg ~ x, d, gaussian())
  .glm_compare(yp ~ x, d, poisson(link = "log"))
  .glm_compare(yb ~ x, d, binomial(link = "logit"))
})

test_that("binomial probit/cloglog/cauchit are handled by Rust", {
  set.seed(105)
  d <- data.frame(x = rnorm(500))
  d$y <- rbinom(500, 1, plogis(-0.4 + 0.8 * d$x))

  for (link in c("probit", "cloglog", "cauchit")) {
    fit <- .glm_compare(y ~ x, d, binomial(link = link), tolerance = 5e-4)
    expect_identical(fit$rust$engine, "faer-glm-irls-fisher-scoring-qr")
  }
})


test_that("custom loglog binomial link is validated against stats::glm", {
  # stats::binomial(link = "loglog") does not exist. Instead construct a valid
  # family object with the inverse-link and derivative functions.
  set.seed(109)
  d <- data.frame(x = rnorm(400))
  d$y <- rbinom(nrow(d), 1L, plogis(0.2 + 0.5 * d$x))

  fam <- stats::binomial(link = "logit")
  fam$link <- "loglog"
  fam$linkfun <- function(mu) -log(-log(mu))
  fam$linkinv <- function(eta) exp(-exp(-eta))
  fam$mu.eta <- function(eta) exp(-eta - exp(-eta))
  fam$valideta <- function(eta) TRUE

  fit <- .glm_compare(y ~ x, d, fam, tolerance = 5e-4)
  expect_identical(fit$rust$engine, "faer-glm-irls-fisher-scoring-qr")
})

test_that("factor and two-column binomial responses preserve trial semantics", {
  set.seed(106)
  d <- data.frame(x = rnorm(300))
  p <- plogis(-0.2 + 0.4 * d$x)
  d$success <- rbinom(nrow(d), size = 10L, prob = p)
  d$failure <- 10L - d$success
  d$factor_y <- factor(rbinom(nrow(d), 1L, p), levels = c(0, 1))

  fit <- .glm_compare(cbind(success, failure) ~ x, d, binomial(), 2e-4)
  expect_equal(unname(fit$prior.weights), rep(10, nrow(d)))
  .glm_compare(factor_y ~ x, d, binomial(), 2e-4)
})

test_that("fixed theta MASS negative binomial uses exact family theta", {
  skip_if_not_installed("MASS")
  set.seed(107)
  theta <- 2.123456789
  d <- data.frame(x = rnorm(500))
  d$y <- rnbinom(nrow(d), size = theta, mu = exp(0.4 + 0.3 * d$x))
  family <- MASS::negative.binomial(theta = theta)

  got <- .glm_compare(y ~ x, d, family, 5e-4)
  expect_identical(got$rust$engine, "faer-glm-irls-fisher-scoring-qr")
  # MASS prints theta to only four decimal places in the family *name*.
  # The fit must obtain the exact theta from the variance closure instead.
  expect_equal(got$rust$coefficients, unname(coef(got)))
  expect_error(
    mrvcr::glm(y ~ x, data = d, family = family, param = theta + 1),
    "differs from family theta"
  )
})

test_that("fixed-power statmod Tweedie with log link uses Rust", {
  skip_if_not_installed("statmod")
  set.seed(108)
  d <- data.frame(x = rnorm(350))
  d$y <- exp(0.2 + 0.3 * d$x + rnorm(nrow(d), sd = 0.25))
  family <- statmod::tweedie(var.power = 1.5, link.power = 0)

  fit <- .glm_compare(y ~ x, d, family, 5e-4)
  expect_identical(fit$rust$engine, "faer-glm-irls-fisher-scoring-qr")
  expect_error(
    mrvcr::glm(y ~ x, data = d, family = family, param = 2),
    "differs from family var.power"
  )
})

test_that("unsupported families and links are rejected by native enums", {
  d <- data.frame(x = 1:8, y = 1:8)
  expect_error(
    mrvcr::glm(y ~ x, data = d, family = stats::Gamma(link = "inverse")),
    "family.*not implemented in Rust"
  )
  expect_error(
    mrvcr::glm(y ~ x, data = d, family = stats::quasipoisson()),
    "family.*not implemented in Rust"
  )
  skip_if_not_installed("statmod")
  fam <- statmod::tweedie(var.power = 1.5, link.power = -0.5)
  expect_error(
    mrvcr::glm(y ~ x, data = d, family = fam),
    "link.*not implemented in Rust"
  )
})

test_that("negative weights and out-of-support response are rejected", {
  d <- data.frame(x = 1:8, y = c(-1, 0, 1, 2, 3, 4, 5, 6))
  expect_error(
    mrvcr::glm(y ~ x, data = d, family = poisson()),
    "outside the support"
  )
  expect_error(
    mrvcr::glm(
      y ~ x,
      data = d,
      family = gaussian(),
      weights = rep(-1, nrow(d))
    ),
    "negative weights"
  )
})
