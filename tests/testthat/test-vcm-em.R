# Tests use independently evaluated likelihood formulas and known estimates;
# no test relies on the Rust likelihood implementation as its own oracle.

.vcm_manual_loglik <- function(y, X, k, theta, pet) {
  S <- Reduce(`+`, Map(function(t, V) t * V, theta, k))
  Q <- solve(S)
  W <- crossprod(X, Q %*% X)
  b <- solve(W, crossprod(X, Q %*% y))
  e <- as.vector(y - X %*% b)
  n <- length(y)
  p <- ncol(X)
  if (pet == "MLE") {
    -0.5 * (n * log(2*pi) + determinant(S, logarithm = TRUE)$modulus[1] +
              drop(crossprod(e, Q %*% e)))
  } else {
    -0.5 * ((n-p) * log(2*pi) +
              determinant(S, logarithm = TRUE)$modulus[1] +
              determinant(W, logarithm = TRUE)$modulus[1] +
              drop(crossprod(e, Q %*% e)))
  }
}

testthat::test_that("identity-only VCM gives closed-form ML and REML variances", {
  set.seed(151)
  n <- 52L
  d <- data.frame(x = seq(-1, 1, length.out = n))
  d$y <- 1 + 1.7*d$x + stats::rnorm(n, sd = 0.7)
  X <- stats::model.matrix(~ x, d)
  rss <- sum(stats::lm.fit(X, d$y)$residuals^2)
  for (pet in c("MLE", "REML")) {
    fit <- mrvcr::vcm(y ~ x, d, kernels = list(resid = diag(n)),
                       pet = pet, method = "EM", theta_init = 1.2,
                       max_iter = 300L, tol = 1e-10)
    expected_var <- rss / if (pet == "MLE") n else (n - ncol(X))
    testthat::expect_equal(unname(fit$variance_components), expected_var, tolerance = 2e-7)
    testthat::expect_equal(unname(stats::coef(fit)),
                            unname(stats::coef(stats::lm(y ~ x, d))), tolerance = 2e-9)
    testthat::expect_equal(unname(stats::fitted(fit)),
                            unname(stats::fitted(stats::lm(y ~ x, d))), tolerance = 2e-9)
    testthat::expect_equal(fit$log_likelihood,
      .vcm_manual_loglik(d$y, X, list(diag(n)), expected_var, pet), tolerance = 1e-7)
    testthat::expect_true(fit$converged)
    testthat::expect_true(all(diff(fit$loglik_history) >= -1e-9))
    testthat::expect_equal(unname(stats::predict(fit, newdata = d[1:5, ])),
      unname(stats::fitted(fit)[1:5]), tolerance = 1e-9)
  }
})

testthat::test_that("low-rank ZZ' kernel supports ML and REML with monotone EM", {
  set.seed(152)
  n <- 36L
  groups <- rep(seq_len(6), each = 6)
  z <- stats::model.matrix(~ 0 + factor(groups))
  K <- tcrossprod(z)
  d <- data.frame(x = stats::rnorm(n))
  d$y <- 2 + 0.3*d$x + stats::rnorm(6)[groups]*0.8 + stats::rnorm(n, sd = 0.45)
  X <- stats::model.matrix(~ x, d)
  for (pet in c("MLE", "REML")) {
    # Convergence of EM can be slow near zero; monotonicity and likelihood
    # evaluation are validated regardless of iteration budget.
    fit <- suppressWarnings(mrvcr::vcm(y ~ x, d,
      kernels = list(group = K, residual = diag(n)),
      pet = pet, theta_init = c(0.5, 0.5),
      tol = 1e-8, max_iter = 450L))
    testthat::expect_equal(unname(fit$kernel_ranks), c(6L, n))
    testthat::expect_true(all(is.finite(fit$variance_components)))
    testthat::expect_true(all(fit$variance_components >= 0))
    testthat::expect_gte(fit$log_likelihood, fit$loglik_history[1L] - 1e-8)
    testthat::expect_true(all(diff(fit$loglik_history) >= -1e-8))
    testthat::expect_equal(fit$log_likelihood,
      .vcm_manual_loglik(d$y, X, list(K, diag(n)),
                         unname(fit$variance_components), pet), tolerance = 1e-7)
    testthat::expect_identical(names(fit$variance_components), c("group", "residual"))
  }
})

testthat::test_that("EM rejects invalid statistical and covariance inputs", {
  d <- data.frame(y = c(2, 3, 4, 5, 6), x = 1:5)
  I <- diag(5)
  testthat::expect_error(mrvcr::vcm(y ~ x, d, list(I), method = "MM"), "Only method")
  testthat::expect_error(mrvcr::vcm(y ~ x, d, list(I), pet = "Ridge"), "pet")
  testthat::expect_error(mrvcr::vcm(y ~ x, d, list(-I)), "positive semidefinite")
  testthat::expect_error(mrvcr::vcm(y ~ x, d, list(matrix(0, 5, 5))), "rank zero")
  testthat::expect_error(mrvcr::vcm(y ~ x, d, list(I), theta_init = 0), "strictly positive")
  testthat::expect_error(mrvcr::vcm(y ~ x, d, list(I[1:4, 1:4])), "5-by-5")
  testthat::expect_error(mrvcr::vcm(y ~ x, d, list(I), max_iter = -1), "positive integer")
  d$y[2] <- NA_real_
  testthat::expect_error(mrvcr::vcm(y ~ x, d, list(I)), "missing")
})
