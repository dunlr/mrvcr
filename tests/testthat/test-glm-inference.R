# GLM inference regression tests. Native Fisher-information covariance is
# opt-in via se=TRUE; base glm-compatible inference uses the existing R QR.

.check_glm_inference <- function(ours, reference, tol = 2e-4,
                                 native_cov = TRUE) {
  so <- summary(ours)
  sr <- summary(reference)

  testthat::expect_true(isTRUE(ours$converged))
  testthat::expect_equal(unname(coef(ours)), unname(coef(reference)), tolerance = tol)
  testthat::expect_equal(ours$df.residual, reference$df.residual)
  # In a glm object, $residuals must be the final IRLS working residuals.
  # Test all public types, including response/pearson/deviance transformations.
  testthat::expect_equal(unname(ours$residuals),
                         unname(reference$residuals), tolerance = tol)
  for (resid_type in c("working", "response", "pearson", "deviance")) {
    testthat::expect_equal(
      unname(stats::residuals(ours, type = resid_type)),
      unname(stats::residuals(reference, type = resid_type)),
      tolerance = tol
    )
  }
  testthat::expect_equal(so$dispersion, sr$dispersion, tolerance = tol)
  testthat::expect_equal(unname(so$coefficients), unname(sr$coefficients),
                         tolerance = tol)
  testthat::expect_equal(unname(stats::vcov(ours)),
                         unname(stats::vcov(reference)), tolerance = tol)
  testthat::expect_equal(unname(so$cov.unscaled),
                         unname(sr$cov.unscaled), tolerance = tol)

  if (native_cov) {
    testthat::expect_true(isTRUE(ours$rust$covariance_available))
    testthat::expect_equal(dim(ours$rust$vcov_unscaled),
                           c(length(coef(ours)), length(coef(ours))))
    testthat::expect_equal(unname(ours$rust$vcov_unscaled),
                           unname(sr$cov.unscaled), tolerance = tol)
    testthat::expect_equal(unname(ours$rust$vcov_unscaled),
                           unname(so$cov.unscaled), tolerance = tol)
  } else {
    testthat::expect_false(isTRUE(ours$rust$covariance_available))
    testthat::expect_null(ours$rust$vcov_unscaled)
  }
  invisible(ours)
}

testthat::test_that("Gaussian GLM matches R covariance, SEs, dispersion and tests", {
  set.seed(7801)
  d <- data.frame(x = rnorm(230), z = runif(230, -2, 3))
  d$y <- 2 + 0.8 * d$x - 0.3 * d$z + rnorm(nrow(d), sd = 1.3)
  ours <- mrvcr::glm(y ~ x + z, data = d, family = gaussian(), se = TRUE)
  reference <- stats::glm(y ~ x + z, data = d, family = gaussian())
  .check_glm_inference(ours, reference, tol = 2e-5)
  testthat::expect_true(summary(ours)$dispersion > 0)
})

testthat::test_that("weighted Poisson with an offset matches inference", {
  set.seed(7802)
  d <- data.frame(x = rnorm(260), off = log(runif(260, 0.7, 2.4)))
  d$w <- runif(nrow(d), 0.5, 3)
  d$w[seq(11, nrow(d), by = 17)] <- 0
  d$y <- rpois(nrow(d), exp(0.3 + 0.45 * d$x + d$off))
  ours <- mrvcr::glm(y ~ x, data = d, family = poisson(),
                     weights = w, offset = off, se = TRUE)
  reference <- stats::glm(y ~ x, data = d, family = poisson(),
                          weights = w, offset = off)
  .check_glm_inference(ours, reference, tol = 2e-4)
  testthat::expect_equal(summary(ours)$dispersion, 1)
  testthat::expect_equal(ours$aic, reference$aic, tolerance = 2e-4)
})

testthat::test_that("binomial logit and probit weighted inference matches R", {
  set.seed(7803)
  d <- data.frame(x = rnorm(330), z = rnorm(330))
  d$w <- sample(1:3, nrow(d), replace = TRUE)
  d$y <- rbinom(nrow(d), 1, plogis(0.1 + 0.6 * d$x - 0.4 * d$z))
  for (link in c("logit", "probit")) {
    family <- binomial(link = link)
    ours <- mrvcr::glm(y ~ x + z, data = d, family = family,
                       weights = w, se = TRUE)
    reference <- stats::glm(y ~ x + z, data = d, family = family,
                            weights = w)
    .check_glm_inference(ours, reference, tol = 8e-4)
    testthat::expect_equal(summary(ours)$dispersion, 1)
  }
})

testthat::test_that("two-column binomial uses trial counts for AIC", {
  set.seed(7804)
  d <- data.frame(x = rnorm(240))
  d$trials <- sample(2:12, nrow(d), replace = TRUE)
  d$success <- rbinom(nrow(d), d$trials, plogis(-0.25 + 0.6 * d$x))
  d$failure <- d$trials - d$success
  d$case_weight <- sample(1:3, nrow(d), replace = TRUE)
  ours <- mrvcr::glm(cbind(success, failure) ~ x, data = d,
                     family = binomial(), weights = case_weight, se = TRUE)
  reference <- stats::glm(cbind(success, failure) ~ x, data = d,
                          family = binomial(), weights = case_weight)
  .check_glm_inference(ours, reference, tol = 2e-4)
  testthat::expect_equal(ours$deviance, reference$deviance, tolerance = 2e-4)
  testthat::expect_equal(ours$aic, reference$aic, tolerance = 2e-4)
})

testthat::test_that("optional Rust covariance is disabled by default", {
  set.seed(7805)
  d <- data.frame(x = rnorm(130))
  d$y <- rpois(nrow(d), exp(0.4 + 0.25 * d$x))
  ours <- mrvcr::glm(y ~ x, data = d, family = poisson())
  reference <- stats::glm(y ~ x, data = d, family = poisson())
  .check_glm_inference(ours, reference, native_cov = FALSE)
  testthat::expect_equal(unname(stats::confint.default(ours)),
                         unname(stats::confint.default(reference)),
                         tolerance = 2e-4)
})

testthat::test_that("fixed-theta negative binomial covariance agrees with R", {
  testthat::skip_if_not_installed("MASS")
  set.seed(7806)
  d <- data.frame(x = runif(220, -2, 2))
  d$y <- rnbinom(nrow(d), mu = exp(0.2 - 0.3 * d$x), size = 4.5)
  fam <- MASS::negative.binomial(theta = 4.5)
  ours <- mrvcr::glm(y ~ x, data = d, family = fam, se = TRUE)
  reference <- stats::glm(y ~ x, data = d, family = fam)
  .check_glm_inference(ours, reference, tol = 4e-4)
})

testthat::test_that("fixed-power Tweedie covariance agrees with R", {
  testthat::skip_if_not_installed("statmod")
  set.seed(7807)
  d <- data.frame(x = rnorm(190))
  d$y <- exp(0.2 + 0.15 * d$x + rnorm(nrow(d), sd = 0.25))
  fam <- statmod::tweedie(var.power = 1.5, link.power = 0)
  ours <- mrvcr::glm(y ~ x, data = d, family = fam, se = TRUE)
  reference <- stats::glm(y ~ x, data = d, family = fam)
  .check_glm_inference(ours, reference, tol = 5e-4)
})

testthat::test_that("native Fisher covariance respects column pivoting", {
  set.seed(7808)
  d <- data.frame(x_small = 0.001 * rnorm(270),
                  x_large = 120 * rnorm(270),
                  x_mid = rnorm(270))
  d$y <- rpois(nrow(d), exp(0.3 + 80 * d$x_small +
                           0.002 * d$x_large + 0.15 * d$x_mid))
  ours <- mrvcr::glm(y ~ x_small + x_large + x_mid, data = d,
                     family = poisson(), se = TRUE)
  reference <- stats::glm(y ~ x_small + x_large + x_mid, data = d,
                          family = poisson())
  .check_glm_inference(ours, reference, tol = 3e-4)
})

testthat::test_that("non-converged GLMs remain explicitly marked", {
  set.seed(7809)
  d <- data.frame(x = rnorm(100))
  d$y <- rpois(nrow(d), exp(0.6 + 0.4 * d$x))
  testthat::expect_warning(
    fit <- mrvcr::glm(y ~ x, data = d, family = poisson(),
                      control = list(maxit = 1L), se = TRUE),
    "did not converge"
  )
  testthat::expect_false(isTRUE(fit$converged))
  testthat::expect_equal(fit$iter, 1L)
})
