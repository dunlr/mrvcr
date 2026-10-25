# Independent checks of the multiplicative MM update. This test computes its
# q_k and trace terms in R, not by calling the Rust optimizer a second time.

.vcm_mm_independent_step <- function(y, X, kernels, theta, pet) {
  covariance <- Reduce(`+`, Map(function(t, K) t * K, theta, kernels))
  inv <- solve(covariance)
  w <- crossprod(X, inv %*% X)
  beta <- solve(w, crossprod(X, inv %*% y))
  residual <- as.vector(y - X %*% beta)
  a <- as.vector(inv %*% residual)
  P <- if (pet == "REML") {
    inv - inv %*% X %*% solve(w, crossprod(X, inv))
  } else {
    inv
  }
  vapply(seq_along(kernels), function(k) {
    q <- drop(crossprod(a, kernels[[k]] %*% a))
    tr <- sum(diag(P %*% kernels[[k]]))
    theta[k] * sqrt(q / tr)
  }, numeric(1))
}

.vcm_mm_fixture <- function() {
  set.seed(1553)
  n <- 30L
  g <- rep(seq_len(6L), each = 5L)
  x <- stats::rnorm(n)
  d <- data.frame(x = x, group = factor(g))
  d$y <- 0.8 + 0.7*x +
    c(-1.2, 1.3, -.7, 1.4, -1.0, .8)[g] + stats::rnorm(n, sd=.23)
  Z <- stats::model.matrix(~ 0 + group, d)
  list(d = d,
       X = stats::model.matrix(~ x, d),
       kernels = list(random = tcrossprod(Z), residual = diag(n)))
}

testthat::test_that("MM matches an independent single-step ML and REML calculation", {
  fixture <- .vcm_mm_fixture()
  d <- fixture$d
  init <- c(0.5, 0.5)
  for (pet in c("MLE", "REML")) {
    expected <- .vcm_mm_independent_step(d$y, fixture$X,
                  fixture$kernels, init, pet)
    fit <- suppressWarnings(mrvcr::vcm(
      y ~ x, d, fixture$kernels, pet = pet, method = "MM",
      theta_init = init, max_iter = 1L, tol = 1e-12
    ))
    testthat::expect_equal(unname(fit$variance_components), expected,
                           tolerance = 5e-9)
    testthat::expect_equal(fit$method, "MM")
    testthat::expect_equal(length(fit$loglik_history), 2L)
    testthat::expect_gte(diff(fit$loglik_history)[1L], -1e-8)
  }
})

testthat::test_that("MM and EM converge to the same ML and REML optima", {
  fixture <- .vcm_mm_fixture()
  d <- fixture$d
  for (pet in c("MLE", "REML")) {
    args <- list(formula = y ~ x, data = d,
                 kernels = fixture$kernels,
                 pet = pet, theta_init = c(.5, .5),
                 max_iter = 250L, tol = 1e-9)
    em <- suppressWarnings(do.call(mrvcr::vcm,
                        c(args, list(method = "EM"))))
    mm <- suppressWarnings(do.call(mrvcr::vcm,
                        c(args, list(method = "MM"))))
    testthat::expect_true(em$converged)
    testthat::expect_true(mm$converged)
    testthat::expect_true(all(diff(mm$loglik_history) >= -1e-8))
    testthat::expect_true(all(mm$variance_components >= 0))
    testthat::expect_equal(unname(mm$variance_components),
                           unname(em$variance_components), tolerance = 1e-6)
    testthat::expect_equal(mm$log_likelihood,
                           em$log_likelihood, tolerance = 1e-8)
    testthat::expect_equal(unname(stats::coef(mm)),
                           unname(stats::coef(em)), tolerance = 1e-7)
  }
})

testthat::test_that("single identity kernel MM solves known ML/REML variance", {
  set.seed(1601)
  n <- 38L
  d <- data.frame(x = seq(-1, 1, length.out = n))
  d$y <- 1 + 1.5*d$x + stats::rnorm(n, sd = .6)
  X <- stats::model.matrix(~ x, d)
  rss <- sum(stats::lm.fit(X, d$y)$residuals^2)
  for (pet in c("MLE", "REML")) {
    fit <- mrvcr::vcm(y ~ x, d, list(residual = diag(n)),
                       pet = pet, method = "MM", theta_init = 2.0,
                       max_iter = 120L, tol = 1e-10)
    target <- rss / if (pet == "MLE") n else (n - ncol(X))
    testthat::expect_true(fit$converged)
    testthat::expect_equal(unname(fit$variance_components),
                           target, tolerance = 1e-7)
    testthat::expect_true(all(diff(fit$loglik_history) >= -1e-8))
  }
})

testthat::test_that("MM rejects unidentifiable REML kernels and unknown algorithms", {
  set.seed(1602)
  n <- 12L
  d <- data.frame(y = stats::rnorm(n))
  # A covariance component confined to the fixed intercept has no
  # restricted-likelihood information; do not divide by a near-zero trace.
  testthat::expect_error(
    mrvcr::vcm(y ~ 1, d,
      kernels = list(intercept = matrix(1, n, n), residual = diag(n)),
      pet = "REML", method = "MM", theta_init = c(1, 1)),
    "unidentifiable"
  )
  testthat::expect_error(
    mrvcr::vcm(y ~ 1, d, kernels = list(diag(n)), method = "NR"),
    "method"
  )
})
