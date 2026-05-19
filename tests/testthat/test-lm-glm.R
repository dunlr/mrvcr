expect_effects_invariants <- function(fit, y_work, tol = 1e-8) {
  effects <- matrix(
    fit$effects,
    nrow = length(y_work) / fit$n_responses,
    ncol = fit$n_responses
  )

  y_mat <- if (is.matrix(y_work)) {
    y_work
  } else {
    matrix(y_work, ncol = 1)
  }

  expect_true(isTRUE(fit$has_effects))
  expect_equal(nrow(effects), nrow(y_mat))
  expect_equal(ncol(effects), ncol(y_mat))

  expect_equal(
    colSums(effects^2),
    colSums(y_mat^2),
    tolerance = tol
  )

  rank <- fit$rank
  if (rank < nrow(effects)) {
    trailing <- effects[(rank + 1L):nrow(effects), , drop = FALSE]
    residuals <- matrix(
      fit$residuals,
      nrow = nrow(y_mat),
      ncol = ncol(y_mat)
    )

    expect_equal(
      colSums(trailing^2),
      colSums(residuals^2),
      tolerance = tol
    )
  }
}

test_that("lm agrees with stats::lm on mtcars in rust compatibility mode", {
  fit_mrvcr <- mrvcr::lm(mpg ~ wt + hp, data = mtcars)
  fit_stats <- stats::lm(mpg ~ wt + hp, data = mtcars)

  expect_s3_class(fit_mrvcr, "mrvcr_lm")
  expect_false(inherits(fit_mrvcr, "lm"))
  expect_equal(fit_mrvcr$compat, "rust")

  expect_equal(unname(coef(fit_mrvcr)), unname(coef(fit_stats)), tolerance = 1e-8)
  expect_equal(unname(fitted(fit_mrvcr)), unname(fitted(fit_stats)), tolerance = 1e-8)
  expect_equal(unname(residuals(fit_mrvcr)), unname(residuals(fit_stats)), tolerance = 1e-8)
  expect_equal(fit_mrvcr$rust$solver_method, "col_piv_qr")
})

test_that("lm handles subset, weights, offset, and missing values like stats::lm", {
  df <- mtcars
  df$w <- seq_len(nrow(df))
  df$off <- 0.05 * df$hp

  expect_equal(
    unname(coef(mrvcr::lm(mpg ~ wt + hp, data = df, subset = cyl != 4))),
    unname(coef(stats::lm(mpg ~ wt + hp, data = df, subset = cyl != 4))),
    tolerance = 1e-8
  )

  expect_equal(
    unname(coef(mrvcr::lm(mpg ~ wt + hp, data = df, weights = w))),
    unname(coef(stats::lm(mpg ~ wt + hp, data = df, weights = w))),
    tolerance = 1e-8
  )

  expect_equal(
    unname(coef(mrvcr::lm(mpg ~ wt, data = df, offset = off))),
    unname(coef(stats::lm(mpg ~ wt, data = df, offset = off))),
    tolerance = 1e-8
  )

  expect_equal(
    unname(coef(mrvcr::lm(mpg ~ wt, data = df, weights = w, offset = off))),
    unname(coef(stats::lm(mpg ~ wt, data = df, weights = w, offset = off))),
    tolerance = 1e-8
  )

  df2 <- mtcars
  df2$hp[1] <- NA_real_

  fit_mrvcr <- mrvcr::lm(mpg ~ wt + hp, data = df2, na.action = stats::na.exclude)
  fit_stats <- stats::lm(mpg ~ wt + hp, data = df2, na.action = stats::na.exclude)

  expect_equal(unname(coef(fit_mrvcr)), unname(coef(fit_stats)), tolerance = 1e-8)
  expect_equal(length(fitted(fit_mrvcr)), length(fitted(fit_stats)))
  expect_equal(length(residuals(fit_mrvcr)), length(residuals(fit_stats)))
})

test_that("direct lm_fit_rust backend agrees with stats::lm.fit", {
  X <- stats::model.matrix(mpg ~ wt + hp, data = mtcars)
  Y <- mtcars$mpg

  fit_rust <- mrvcr:::lm_fit_rust(Y, X, NULL, "OLS", "qr", 1e-7, TRUE, FALSE)
  fit_stats <- stats::lm.fit(x = X, y = Y)

  expect_equal(fit_rust$solver_method, "col_piv_qr")
  expect_equal(unname(fit_rust$coefficients), unname(fit_stats$coefficients), tolerance = 1e-8)
  expect_equal(unname(fit_rust$fitted_values), unname(fit_stats$fitted.values), tolerance = 1e-8)
  expect_equal(unname(fit_rust$residuals), unname(fit_stats$residuals), tolerance = 1e-8)
  expect_equal(fit_rust$n_responses, 1L)
})

test_that("method = householder_qr keeps unpivoted QR backend available", {
  X <- stats::model.matrix(mpg ~ wt + hp, data = mtcars)
  Y <- mtcars$mpg

  fit_rust <- mrvcr:::lm_fit_rust(Y, X, NULL, "OLS", "householder_qr", 1e-7, TRUE, FALSE)
  fit_stats <- stats::lm.fit(x = X, y = Y)

  expect_equal(fit_rust$solver_method, "householder_qr")
  expect_equal(unname(fit_rust$coefficients), unname(fit_stats$coefficients), tolerance = 1e-8)
})

test_that("direct lm_fit_rust with offset agrees with stats::lm.fit", {
  X <- stats::model.matrix(mpg ~ wt, data = mtcars)
  Y <- mtcars$mpg
  offset <- 0.05 * mtcars$hp

  fit_rust <- mrvcr:::lm_fit_rust(Y, X, offset, "OLS", "qr", 1e-7, TRUE, FALSE)
  fit_stats <- stats::lm.fit(x = X, y = Y, offset = offset)

  expect_equal(unname(fit_rust$coefficients), unname(fit_stats$coefficients), tolerance = 1e-8)
  expect_equal(unname(fit_rust$fitted_values), unname(fit_stats$fitted.values), tolerance = 1e-8)
  expect_equal(unname(fit_rust$residuals), unname(fit_stats$residuals), tolerance = 1e-8)
})

test_that("direct wlm_fit_rust backend agrees with stats::lm.wfit", {
  X <- stats::model.matrix(mpg ~ wt + hp, data = mtcars)
  Y <- mtcars$mpg
  W <- seq_len(nrow(mtcars))

  fit_rust <- mrvcr:::wlm_fit_rust(Y, X, W, NULL, "WLS", "qr", 1e-7, TRUE, FALSE)
  fit_stats <- stats::lm.wfit(x = X, y = Y, w = W)

  expect_equal(fit_rust$solver_method, "col_piv_qr")
  expect_equal(unname(fit_rust$coefficients), unname(fit_stats$coefficients), tolerance = 1e-8)
  expect_equal(unname(fit_rust$fitted_values), unname(fit_stats$fitted.values), tolerance = 1e-8)
  expect_equal(unname(fit_rust$residuals), unname(fit_stats$residuals), tolerance = 1e-8)
})

test_that("direct wlm_fit_rust with zero weights agrees with stats::lm.wfit", {
  X <- stats::model.matrix(mpg ~ wt + hp, data = mtcars)
  Y <- mtcars$mpg
  W <- rep(1, nrow(mtcars))
  W[1:5] <- 0

  fit_rust <- mrvcr:::wlm_fit_rust(Y, X, W, NULL, "WLS", "qr", 1e-7, TRUE, FALSE)
  fit_stats <- stats::lm.wfit(x = X, y = Y, w = W)

  expect_equal(unname(fit_rust$coefficients), unname(fit_stats$coefficients), tolerance = 1e-8)
  expect_equal(unname(fit_rust$fitted_values), unname(fit_stats$fitted.values), tolerance = 1e-8)
  expect_equal(unname(fit_rust$residuals), unname(fit_stats$residuals), tolerance = 1e-8)
})

test_that("lm_fit_rust cholesky agrees with qr on well-conditioned data", {
  set.seed(1)

  n <- 1000
  p <- 10

  X_raw <- matrix(rnorm(n * p), nrow = n)
  X <- cbind("(Intercept)" = 1, X_raw)
  beta <- seq_len(p + 1) / (p + 1)
  Y <- drop(X %*% beta + rnorm(n, sd = 0.1))

  fit_qr <- mrvcr:::lm_fit_rust(Y, X, NULL, "OLS", "qr", 1e-7, TRUE, FALSE)
  fit_chol <- mrvcr:::lm_fit_rust(Y, X, NULL, "OLS", "cholesky", 1e-7, TRUE, FALSE)

  expect_equal(unname(fit_chol$coefficients), unname(fit_qr$coefficients), tolerance = 1e-8)
  expect_false(isTRUE(fit_chol$has_effects))
  expect_equal(length(fit_chol$effects), 0L)
})

test_that("lm_fit_rust cholesky errors on exactly singular design", {
  set.seed(1)

  n <- 1000
  x1 <- rnorm(n)
  X <- cbind("(Intercept)" = 1, x1 = x1, x1_copy = x1)
  Y <- 1 + 2 * x1 + rnorm(n, sd = 0.1)

  expect_error(
    mrvcr:::lm_fit_rust(Y, X, NULL, "OLS", "cholesky", 1e-7, TRUE, FALSE),
    "Cholesky failed|positive definite|singular"
  )
})

test_that("lm_fit_rust qr detects exact rank deficiency", {
  set.seed(1)

  n <- 100
  x1 <- rnorm(n)
  X <- cbind("(Intercept)" = 1, x1 = x1, x1_copy = x1)
  Y <- 1 + 2 * x1 + rnorm(n, sd = 0.1)

  fit_rust <- mrvcr:::lm_fit_rust(Y, X, NULL, "OLS", "qr", 1e-7, TRUE, FALSE)
  fit_stats <- stats::lm.fit(X, Y)

  expect_lt(fit_rust$rank, ncol(X))
  expect_equal(fit_rust$rank, fit_stats$rank)
  expect_true(isTRUE(fit_rust$pivoted))
  expect_true(any(is.na(fit_rust$coefficients)))
  expect_equal(fit_rust$solver_method, "col_piv_qr")
})

test_that("lm_fit_rust qr respects singular.ok = FALSE", {
  set.seed(1)

  n <- 100
  x1 <- rnorm(n)
  X <- cbind("(Intercept)" = 1, x1 = x1, x1_copy = x1)
  Y <- 1 + 2 * x1 + rnorm(n, sd = 0.1)

  expect_error(
    mrvcr:::lm_fit_rust(Y, X, NULL, "OLS", "qr", 1e-7, FALSE, FALSE),
    "singular fit"
  )
})

test_that("lm_fit_rust qr places aliased coefficients like stats::lm.fit", {
  set.seed(1)

  n <- 100
  x1 <- rnorm(n)
  X <- cbind("(Intercept)" = 1, x1 = x1, x1_copy = x1)
  Y <- 1 + 2 * x1 + rnorm(n, sd = 0.1)

  fit_rust <- mrvcr:::lm_fit_rust(Y, X, NULL, "OLS", "qr", 1e-7, TRUE, FALSE)
  fit_stats <- stats::lm.fit(X, Y)

  expect_equal(fit_rust$rank, fit_stats$rank)
  expect_true(isTRUE(fit_rust$pivoted))
  expect_equal(length(fit_rust$pivot), ncol(X))
  expect_true(all(sort(fit_rust$pivot) == seq_len(ncol(X))))

  expect_equal(
    unname(is.na(fit_rust$coefficients)),
    unname(is.na(fit_stats$coefficients))
  )
})

test_that("wlm_fit_rust qr places aliased coefficients like stats::lm.wfit", {
  set.seed(1)

  n <- 100
  x1 <- rnorm(n)
  X <- cbind("(Intercept)" = 1, x1 = x1, x1_copy = x1)
  Y <- 1 + 2 * x1 + rnorm(n, sd = 0.1)
  W <- seq_len(n)

  fit_rust <- mrvcr:::wlm_fit_rust(Y, X, W, NULL, "WLS", "qr", 1e-7, TRUE, FALSE)
  fit_stats <- stats::lm.wfit(X, Y, W)

  expect_equal(fit_rust$rank, fit_stats$rank)
  expect_true(isTRUE(fit_rust$pivoted))
  expect_equal(length(fit_rust$pivot), ncol(X))
  expect_true(all(sort(fit_rust$pivot) == seq_len(ncol(X))))

  expect_equal(
    unname(is.na(fit_rust$coefficients)),
    unname(is.na(fit_stats$coefficients))
  )
})

test_that("lm_fit_rust supports multivariate response Y", {
  set.seed(1)

  n <- 200
  p <- 5
  r <- 3

  X_raw <- matrix(rnorm(n * p), nrow = n)
  X <- cbind("(Intercept)" = 1, X_raw)

  B <- matrix(rnorm((p + 1) * r), nrow = p + 1, ncol = r)
  Y <- X %*% B + matrix(rnorm(n * r, sd = 0.1), nrow = n)

  fit_rust <- mrvcr:::lm_fit_rust(Y, X, NULL, "OLS", "qr", 1e-7, TRUE, FALSE)
  fit_stats <- stats::lm.fit(X, Y)

  coef_rust <- matrix(fit_rust$coefficients, nrow = ncol(X), ncol = ncol(Y))
  fitted_rust <- matrix(fit_rust$fitted_values, nrow = nrow(X), ncol = ncol(Y))
  residuals_rust <- matrix(fit_rust$residuals, nrow = nrow(X), ncol = ncol(Y))

  expect_equal(fit_rust$n_responses, ncol(Y))
  expect_equal(unname(coef_rust), unname(fit_stats$coefficients), tolerance = 1e-8)
  expect_equal(unname(fitted_rust), unname(fit_stats$fitted.values), tolerance = 1e-8)
  expect_equal(unname(residuals_rust), unname(fit_stats$residuals), tolerance = 1e-8)
})

test_that("wlm_fit_rust supports multivariate response Y", {
  set.seed(1)

  n <- 200
  p <- 5
  r <- 3

  X_raw <- matrix(rnorm(n * p), nrow = n)
  X <- cbind("(Intercept)" = 1, X_raw)

  B <- matrix(rnorm((p + 1) * r), nrow = p + 1, ncol = r)
  Y <- X %*% B + matrix(rnorm(n * r, sd = 0.1), nrow = n)
  W <- seq_len(n)

  fit_rust <- mrvcr:::wlm_fit_rust(Y, X, W, NULL, "WLS", "qr", 1e-7, TRUE, FALSE)
  fit_stats <- stats::lm.wfit(X, Y, W)

  coef_rust <- matrix(fit_rust$coefficients, nrow = ncol(X), ncol = ncol(Y))
  fitted_rust <- matrix(fit_rust$fitted_values, nrow = nrow(X), ncol = ncol(Y))
  residuals_rust <- matrix(fit_rust$residuals, nrow = nrow(X), ncol = ncol(Y))

  expect_equal(fit_rust$n_responses, ncol(Y))
  expect_equal(unname(coef_rust), unname(fit_stats$coefficients), tolerance = 1e-8)
  expect_equal(unname(fitted_rust), unname(fit_stats$fitted.values), tolerance = 1e-8)
  expect_equal(unname(residuals_rust), unname(fit_stats$residuals), tolerance = 1e-8)
})

test_that("lm_fit_rust cholesky supports multivariate response Y", {
  set.seed(2)

  n <- 500
  p <- 10
  r <- 4

  X_raw <- matrix(rnorm(n * p), nrow = n)
  X <- cbind("(Intercept)" = 1, X_raw)

  B <- matrix(rnorm((p + 1) * r), nrow = p + 1, ncol = r)
  Y <- X %*% B + matrix(rnorm(n * r, sd = 0.1), nrow = n)

  fit_qr <- mrvcr:::lm_fit_rust(Y, X, NULL, "OLS", "qr", 1e-7, TRUE, FALSE)
  fit_chol <- mrvcr:::lm_fit_rust(Y, X, NULL, "OLS", "cholesky", 1e-7, TRUE, FALSE)

  expect_equal(fit_chol$coefficients, fit_qr$coefficients, tolerance = 1e-8)
  expect_equal(fit_chol$fitted_values, fit_qr$fitted_values, tolerance = 1e-8)
})

test_that("calc_se errors for multivariate response", {
  set.seed(1)

  X <- cbind(1, matrix(rnorm(100 * 3), nrow = 100))
  Y <- matrix(rnorm(100 * 2), nrow = 100)

  expect_error(
    mrvcr:::lm_fit_rust(Y, X, NULL, "OLS", "qr", 1e-7, TRUE, TRUE),
    "multi-response"
  )
})

test_that("lm_fit_rust returns valid effects for qr backend", {
  X <- stats::model.matrix(mpg ~ wt + hp, data = mtcars)
  Y <- mtcars$mpg

  fit_rust <- mrvcr:::lm_fit_rust(
    Y, X, NULL, "OLS", "qr", 1e-7, TRUE, FALSE
  )

  expect_equal(fit_rust$solver_method, "col_piv_qr")
  expect_effects_invariants(fit_rust, Y)
})

test_that("lm_fit_rust returns effects for householder_qr backend", {
  X <- stats::model.matrix(mpg ~ wt + hp, data = mtcars)
  Y <- mtcars$mpg

  fit_rust <- mrvcr:::lm_fit_rust(Y, X, NULL, "OLS", "householder_qr", 1e-7, TRUE, FALSE)

  expect_true(isTRUE(fit_rust$has_effects))
  expect_equal(length(fit_rust$effects), length(Y))
  expect_equal(fit_rust$solver_method, "householder_qr")
})

test_that("lm_fit_rust returns valid matrix effects for multivariate response", {
  set.seed(1)

  n <- 200
  p <- 5
  r <- 3

  X_raw <- matrix(rnorm(n * p), nrow = n)
  X <- cbind("(Intercept)" = 1, X_raw)

  B <- matrix(rnorm((p + 1) * r), nrow = p + 1, ncol = r)
  Y <- X %*% B + matrix(rnorm(n * r, sd = 0.1), nrow = n)

  fit_rust <- mrvcr:::lm_fit_rust(
    Y, X, NULL, "OLS", "qr", 1e-7, TRUE, FALSE
  )

  expect_equal(fit_rust$solver_method, "col_piv_qr")
  expect_effects_invariants(fit_rust, Y)
})

test_that("lm_fit_rust cholesky does not compute effects", {
  X <- stats::model.matrix(mpg ~ wt + hp, data = mtcars)
  Y <- mtcars$mpg

  fit_rust <- mrvcr:::lm_fit_rust(Y, X, NULL, "OLS", "cholesky", 1e-7, TRUE, FALSE)

  expect_false(isTRUE(fit_rust$has_effects))
  expect_equal(length(fit_rust$effects), 0L)
})

test_that("glm gaussian agrees with stats::glm", {
  fit_mrvcr <- mrvcr::glm(mpg ~ wt + hp, data = mtcars, family = gaussian)
  fit_stats <- stats::glm(mpg ~ wt + hp, data = mtcars, family = gaussian)

  expect_s3_class(fit_mrvcr, "mrvcr_glm")
  expect_s3_class(fit_mrvcr, "glm")
  expect_s3_class(fit_mrvcr, "lm")

  expect_equal(unname(coef(fit_mrvcr)), unname(coef(fit_stats)), tolerance = 1e-8)
  expect_equal(fit_mrvcr$deviance, fit_stats$deviance, tolerance = 1e-8)
  expect_equal(fit_mrvcr$null.deviance, fit_stats$null.deviance, tolerance = 1e-8)
})

test_that("glm gaussian identity agrees with mrvcr lm", {
  fit_lm <- mrvcr::lm(mpg ~ wt + hp, data = mtcars)
  fit_glm <- mrvcr::glm(mpg ~ wt + hp, data = mtcars, family = gaussian)

  expect_equal(unname(coef(fit_glm)), unname(coef(fit_lm)), tolerance = 1e-8)
})

test_that("glm poisson log is close to stats::glm", {
  df <- data.frame(
    y = c(1, 3, 2, 5, 7, 8, 12, 10, 15, 18),
    x = seq_len(10)
  )

  fit_mrvcr <- mrvcr::glm(y ~ x, data = df, family = poisson(link = "log"))
  fit_stats <- stats::glm(y ~ x, data = df, family = poisson(link = "log"))

  expect_equal(unname(coef(fit_mrvcr)), unname(coef(fit_stats)), tolerance = 1e-4)
})

test_that("glm handles weights and offset like stats::glm for gaussian", {
  df <- mtcars
  df$w <- seq_len(nrow(df))
  df$off <- 0.05 * df$hp

  fit_mrvcr <- mrvcr::glm(mpg ~ wt, data = df, family = gaussian, weights = w, offset = off)
  fit_stats <- stats::glm(mpg ~ wt, data = df, family = gaussian, weights = w, offset = off)

  expect_equal(unname(coef(fit_mrvcr)), unname(coef(fit_stats)), tolerance = 1e-8)
  expect_equal(fit_mrvcr$deviance, fit_stats$deviance, tolerance = 1e-8)
})

test_that("glm rejects negative weights", {
  df <- mtcars
  df$w <- rep(1, nrow(df))
  df$w[1] <- -1

  expect_error(
    mrvcr::glm(mpg ~ wt + hp, data = df, family = gaussian, weights = w),
    "negative weights"
  )
})

test_that("invalid lm method errors cleanly", {
  expect_error(
    mrvcr::lm(mpg ~ wt, data = mtcars, method = "bad_method")
  )
})

test_that("lm_fit_rust returns valid effects for householder_qr backend", {
  X <- stats::model.matrix(mpg ~ wt + hp, data = mtcars)
  Y <- mtcars$mpg

  fit_rust <- mrvcr:::lm_fit_rust(
    Y, X, NULL, "OLS", "householder_qr", 1e-7, TRUE, FALSE
  )

  expect_equal(fit_rust$solver_method, "householder_qr")
  expect_effects_invariants(fit_rust, Y)
})


test_that("compat = rust does not inherit from lm", {
  fit <- mrvcr::lm(mpg ~ wt + hp, data = mtcars, compat = "rust")

  expect_s3_class(fit, "mrvcr_lm")
  expect_false(inherits(fit, "lm"))
  expect_equal(fit$compat, "rust")
  expect_null(fit$qr)
})

test_that("compat = stats inherits from lm and has stats-compatible qr", {
  fit <- mrvcr::lm(mpg ~ wt + hp, data = mtcars, compat = "stats")
  ref <- stats::lm(mpg ~ wt + hp, data = mtcars)

  expect_s3_class(fit, "mrvcr_lm")
  expect_s3_class(fit, "lm")
  expect_equal(fit$compat, "stats")
  expect_s3_class(fit$qr, "qr")

  expect_equal(unname(fit$effects), unname(ref$effects), tolerance = 1e-8)
})

test_that("method = dqrls is recognized but not implemented", {
  expect_error(
    mrvcr::lm(mpg ~ wt + hp, data = mtcars, method = "dqrls"),
    "not implemented"
  )

  expect_error(
    mrvcr::lm(mpg ~ wt + hp, data = mtcars, method = "r_qr"),
    "not implemented"
  )
})

test_that("mrvcr::lm supports multivariate response in rust compatibility mode", {
  fit <- mrvcr::lm(cbind(mpg, disp) ~ wt + hp, data = mtcars, compat = "rust")

  expect_s3_class(fit, "mrvcr_mlm")
  expect_s3_class(fit, "mrvcr_lm")
  expect_false(inherits(fit, "mlm"))
  expect_false(inherits(fit, "lm"))

  expect_true(is.matrix(coef(fit)))
  expect_equal(ncol(coef(fit)), 2L)
  expect_equal(ncol(fitted(fit)), 2L)
  expect_equal(ncol(residuals(fit)), 2L)
})

test_that("mrvcr::lm supports multivariate response in stats compatibility mode", {
  fit <- mrvcr::lm(cbind(mpg, disp) ~ wt + hp, data = mtcars, compat = "stats")
  ref <- stats::lm(cbind(mpg, disp) ~ wt + hp, data = mtcars)

  expect_s3_class(fit, "mrvcr_mlm")
  expect_s3_class(fit, "mlm")
  expect_s3_class(fit, "lm")

  expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-8)
  expect_equal(unname(fitted(fit)), unname(fitted(ref)), tolerance = 1e-8)
  expect_equal(unname(residuals(fit)), unname(residuals(ref)), tolerance = 1e-8)
})

test_that("compat = stats provides lm inheritance for fallback S3 methods", {
  fit_mrvcr <- mrvcr::lm(mpg ~ wt + hp, data = mtcars, compat = "stats")
  fit_stats <- stats::lm(mpg ~ wt + hp, data = mtcars)

  expect_s3_class(fit_mrvcr, "mrvcr_lm")
  expect_s3_class(fit_mrvcr, "lm")
  expect_equal(unname(fit_mrvcr$effects), unname(fit_stats$effects), tolerance = 1e-8)
  expect_s3_class(fit_mrvcr$qr, "qr")
})
