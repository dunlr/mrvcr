# Regression tests for the native Rust GLM IRLS numerical safeguards.
# These require a rebuilt installed mrvcr DLL (restart R on Windows).

.test_glm_against_stats <- function(formula, data, family, weights = NULL,
                                    offset = NULL, tol = 3e-5) {
  # Formula interfaces use nonstandard evaluation for weights/offset.
  # Store supplied test vectors as columns in the model data. This avoids
  # a forwarded `weights = weights` expression being resolved as the
  # stats::weights() function inside model.frame().
  dat <- data
  if (!is.null(weights)) {
    stopifnot(length(weights) == nrow(dat))
    dat[[".mrvcr_test_prior_w"]] <- weights
  }
  if (!is.null(offset)) {
    stopifnot(length(offset) == nrow(dat))
    dat[[".mrvcr_test_eta_offset"]] <- offset
  }

  if (is.null(weights) && is.null(offset)) {
    ours <- mrvcr::glm(formula, data = dat, family = family)
    reference <- stats::glm(formula, data = dat, family = family)
  } else if (is.null(offset)) {
    ours <- mrvcr::glm(formula, data = dat, family = family,
                       weights = .mrvcr_test_prior_w)
    reference <- stats::glm(formula, data = dat, family = family,
                            weights = .mrvcr_test_prior_w)
  } else if (is.null(weights)) {
    ours <- mrvcr::glm(formula, data = dat, family = family,
                       offset = .mrvcr_test_eta_offset)
    reference <- stats::glm(formula, data = dat, family = family,
                            offset = .mrvcr_test_eta_offset)
  } else {
    ours <- mrvcr::glm(formula, data = dat, family = family,
                       weights = .mrvcr_test_prior_w,
                       offset = .mrvcr_test_eta_offset)
    reference <- stats::glm(formula, data = dat, family = family,
                            weights = .mrvcr_test_prior_w,
                            offset = .mrvcr_test_eta_offset)
  }
  testthat::expect_true(isTRUE(ours$converged))
  testthat::expect_equal(unname(coef(ours)), unname(coef(reference)),
                         tolerance = tol)
  testthat::expect_equal(unname(fitted(ours)), unname(fitted(reference)),
                         tolerance = tol)
  testthat::expect_equal(ours$deviance, reference$deviance,
                         tolerance = tol)
  testthat::expect_equal(ours$rust$deviance, ours$deviance,
                         tolerance = 5e-6)
  testthat::expect_true(ours$rust$step_halvings >= 0)
  ours
}

testthat::test_that("weighted Gaussian regression preserves fit and deviance", {
  set.seed(3201)
  d <- data.frame(x = rnorm(220))
  d$y <- 0.7 + 2.0 * d$x + rnorm(nrow(d), sd = 0.8)
  w <- runif(nrow(d), 0.2, 3.5)
  w[seq(1L, nrow(d), 19L)] <- 0
  .test_glm_against_stats(y ~ x, d, gaussian(), weights = w)
})

testthat::test_that("Poisson log link with offsets and zero prior weights", {
  set.seed(3202)
  d <- data.frame(x = runif(250, -1, 1), exposure = runif(250, 0.5, 2.0))
  d$y <- rpois(nrow(d), d$exposure * exp(0.5 - 0.4 * d$x))
  w <- rep(1, nrow(d))
  w[seq(3L, nrow(d), 13L)] <- 0
  .test_glm_against_stats(y ~ x, d, poisson(), weights = w,
                          offset = log(d$exposure), tol = 1e-4)
})

testthat::test_that("binomial probit and logit support prior weights", {
  set.seed(3203)
  d <- data.frame(x = rnorm(350))
  d$y <- rbinom(nrow(d), 1, plogis(-0.3 + 0.5 * d$x))
  w <- sample(1:4, nrow(d), replace = TRUE)
  for (link in c("logit", "probit")) {
    .test_glm_against_stats(y ~ x, d, binomial(link = link),
                            weights = w, tol = 5e-4)
  }
})

testthat::test_that("damped IRLS step avoids large Poisson overshoot", {
  # Starting at beta=0, the first undamped intercept-only Poisson step is
  # ~49.5. Without a line search, exp(49.5) badly overshoots the optimum.
  d <- data.frame(y = c(rep(1L, 15), rep(100L, 15)))
  fit <- mrvcr::glm(y ~ 1, data = d, family = poisson(),
                    control = list(maxit = 100L, epsilon = 1e-8))
  testthat::expect_true(fit$converged)
  testthat::expect_gt(fit$rust$step_halvings, 0L)
  testthat::expect_equal(unname(coef(fit)), log(mean(d$y)),
                         tolerance = 1e-6)
  testthat::expect_equal(fit$rust$deviance, fit$deviance,
                         tolerance = 1e-6)
})

testthat::test_that("nonconvergence is explicitly reported", {
  set.seed(3204)
  d <- data.frame(x = rnorm(70))
  d$y <- 3 + 0.8 * d$x + rnorm(nrow(d))
  testthat::expect_warning(
    fit <- mrvcr::glm(y ~ x, data = d, family = gaussian(),
                      control = list(maxit = 1L)),
    "did not converge"
  )
  testthat::expect_false(isTRUE(fit$converged))
  testthat::expect_equal(fit$iter, 1L)
})

testthat::test_that("singular.ok FALSE rejects duplicated GLM predictors", {
  set.seed(3205)
  d <- data.frame(x = rnorm(60))
  d$z <- d$x
  d$y <- rpois(nrow(d), exp(0.2 + d$x / 3))
  testthat::expect_error(
    mrvcr::glm(y ~ x + z, data = d, family = poisson(),
               singular.ok = FALSE),
    "singular fit"
  )
})

testthat::test_that("non-finite offset is rejected by Rust", {
  d <- data.frame(x = seq(-1, 1, length.out = 20), y = rep(1L, 20))
  d$off <- rep(0, nrow(d))
  d$off[3] <- Inf
  testthat::expect_error(
    mrvcr::glm(y ~ x, data = d, family = poisson(), offset = off),
    "offset contains a non-finite value"
  )
})

testthat::test_that("fixed-theta negative binomial Rust deviance agrees with R", {
  testthat::skip_if_not_installed("MASS")
  set.seed(3206)
  d <- data.frame(x = runif(300, -1, 1))
  d$y <- rnbinom(nrow(d), mu = exp(0.2 + 0.2 * d$x), size = 3.2)
  .test_glm_against_stats(y ~ x, d, MASS::negative.binomial(theta = 3.2),
                          tol = 2e-4)
})

testthat::test_that("fixed-power Tweedie Rust deviance agrees with R", {
  testthat::skip_if_not_installed("statmod")
  set.seed(3207)
  d <- data.frame(x = rnorm(180))
  d$y <- exp(0.4 + 0.2 * d$x + rnorm(nrow(d), sd = 0.15))
  .test_glm_against_stats(y ~ x, d,
                          statmod::tweedie(var.power = 1.5, link.power = 0),
                          tol = 3e-4)
})
