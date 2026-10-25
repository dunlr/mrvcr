# Iteration 5C: independently check profile score/KKT conditions and starts.

.vcm_diag_fixture <- function() {
  set.seed(506)
  n <- 28L
  group <- factor(rep(seq_len(7L), each = 4L))
  x <- stats::rnorm(n)
  d <- data.frame(x = x, group = group)
  d$y <- 0.4 + 0.9 * x + stats::rnorm(7L)[as.integer(group)] * 0.65 +
    stats::rnorm(n, sd = 0.3)
  Z <- stats::model.matrix(~ 0 + group, d)
  list(d = d, X = stats::model.matrix(~ x, d),
       K = list(group = tcrossprod(Z), residual = diag(n)))
}

.vcm_independent_score <- function(y, X, K, theta, pet) {
  S <- Reduce(`+`, Map(function(t, A) t * A, theta, K))
  A <- solve(S)
  W <- crossprod(X, A %*% X)
  b <- solve(W, crossprod(X, A %*% y))
  a <- as.vector(A %*% (y - X %*% b))
  P <- if (pet == "REML") {
    A - A %*% X %*% solve(W, crossprod(X, A))
  } else A
  q <- vapply(K, function(B) drop(crossprod(a, B %*% a)), numeric(1))
  trace <- vapply(K, function(B) sum(diag(P %*% B)), numeric(1))
  list(score = (q - trace) / 2, q = q, trace = trace)
}

.vcm_independent_likelihood <- function(y, X, K, theta, pet) {
  S <- Reduce(`+`, Map(function(t, A) t * A, theta, K))
  A <- solve(S)
  W <- crossprod(X, A %*% X)
  b <- solve(W, crossprod(X, A %*% y))
  r <- as.vector(y - X %*% b)
  n <- length(y)
  p <- ncol(X)
  dimension <- if (pet == "REML") n - p else n
  det_w <- if (pet == "REML")
    as.numeric(determinant(W, logarithm = TRUE)$modulus) else 0
  -0.5 * (dimension * log(2 * pi) +
          as.numeric(determinant(S, logarithm = TRUE)$modulus) +
          det_w + drop(crossprod(r, A %*% r)))
}

testthat::test_that("VCM raw ML/REML scores match independently computed derivatives", {
  f <- .vcm_diag_fixture()
  d <- f$d
  for (pet in c("MLE", "REML")) {
    for (method in c("EM", "MM")) {
      fit <- suppressWarnings(mrvcr::vcm(y ~ x, d, f$K,
        theta_init = c(group = 0.5, residual = 0.25),
        pet = pet, method = method, max_iter = 1L, tol = 1e-12))
      theta <- unname(fit$variance_components)
      analytic <- .vcm_independent_score(d$y, f$X, f$K, theta, pet)
      testthat::expect_equal(unname(fit$score), unname(analytic$score), tolerance = 2e-7)
      # Finite-difference score computed from a completely separate likelihood.
      for (k in seq_along(theta)) {
        h <- 1e-5 * max(theta[k], 0.1)
        plus <- minus <- theta
        plus[k] <- plus[k] + h
        minus[k] <- minus[k] - h
        numeric_score <- (
          .vcm_independent_likelihood(d$y, f$X, f$K, plus, pet) -
          .vcm_independent_likelihood(d$y, f$X, f$K, minus, pet)
        ) / (2 * h)
        testthat::expect_equal(unname(fit$score[k]), numeric_score, tolerance = 2e-5)
      }
      threshold <- 1e-8 * (1 + max(theta))
      boundary <- theta <= threshold
      error <- ifelse(boundary, pmax(analytic$score, 0), abs(analytic$score))
      scaled <- error / (1 + 0.5 * (abs(analytic$q) + abs(analytic$trace)))
      testthat::expect_equal(fit$kkt_violation, max(error), tolerance = 1e-7)
      testthat::expect_equal(fit$kkt_violation_scaled, max(scaled), tolerance = 1e-7)
      testthat::expect_identical(unname(fit$boundary_components), boundary)
      testthat::expect_identical(fit$stop_rule, "parameter")
    }
  }
})

testthat::test_that("initializers respect kernel scales and remain strictly positive", {
  f <- .vcm_diag_fixture()
  # A scaled residual kernel should receive a proportionally smaller
  # raw variance coefficient with trace-scaled starts.
  K <- f$K
  K$residual <- 100 * K$residual
  equal <- suppressWarnings(mrvcr::vcm(y ~ x, f$d, K,
    init = "equal", max_iter = 1L))
  scaled <- suppressWarnings(mrvcr::vcm(y ~ x, f$d, K,
    init = "trace", max_iter = 1L))
  moments <- suppressWarnings(mrvcr::vcm(y ~ x, f$d, K,
    init = "moments", max_iter = 1L))
  testthat::expect_identical(equal$initialization, "equal")
  testthat::expect_equal(unname(equal$theta_initial[1]),
                         unname(equal$theta_initial[2]))
  testthat::expect_lt(scaled$theta_initial[["residual"]],
                      equal$theta_initial[["residual"]])
  testthat::expect_true(all(is.finite(moments$theta_initial)))
  testthat::expect_true(all(moments$theta_initial > 0))
  testthat::expect_identical(moments$initialization, "moments")
  testthat::expect_identical(names(scaled$theta_initial), c("group", "residual"))
  set.seed(507)
  r1 <- suppressWarnings(mrvcr::vcm(y ~ x, f$d, K, init = "random", max_iter = 1L))
  set.seed(507)
  r2 <- suppressWarnings(mrvcr::vcm(y ~ x, f$d, K, init = "random", max_iter = 1L))
  testthat::expect_equal(r1$theta_initial, r2$theta_initial)
})

testthat::test_that("warm starts and named starts use kernel names", {
  f <- .vcm_diag_fixture()
  initial <- c(group = .3, residual = .5)
  fit <- suppressWarnings(mrvcr::vcm(y ~ x, f$d, f$K, pet = "MLE",
    method = "EM", theta_init = rev(initial), max_iter = 3L))
  testthat::expect_equal(fit$theta_initial, initial)
  next_fit <- suppressWarnings(mrvcr::vcm(y ~ x, f$d, f$K, pet = "MLE",
    method = "MM", theta_init = fit, max_iter = 3L))
  testthat::expect_equal(next_fit$theta_initial, fit$variance_components)
  testthat::expect_identical(next_fit$initialization, "warm_start")
  testthat::expect_error(mrvcr::vcm(y ~ x, f$d, f$K,
    theta_init = c(a = 0.5, b = 0.25)), "kernel names")
})

testthat::test_that("KKT stopping is distinct from small parameter steps", {
  f <- .vcm_diag_fixture()
  # Parameter-only convergence after one step is deliberately permissive.
  loose <- mrvcr::vcm(y ~ x, f$d, f$K,
    theta_init = c(.1, 2), pet = "REML", method = "EM",
    tol = 1e2, kkt_tol = 1e-7, stop_rule = "parameter", max_iter = 1L)
  testthat::expect_true(loose$converged)
  testthat::expect_true(loose$converged_parameters)
  testthat::expect_false(loose$converged_kkt)
  testthat::expect_gt(loose$kkt_violation_scaled, 1e-7)
  # Combined stopping must not claim success based solely on tiny steps.
  both <- suppressWarnings(mrvcr::vcm(y ~ x, f$d, f$K,
    theta_init = c(.1, 2), pet = "REML", method = "EM",
    tol = 1e2, kkt_tol = 1e-7, stop_rule = "both", max_iter = 3L))
  testthat::expect_gte(both$iterations, loose$iterations)
  testthat::expect_true(all(diff(both$loglik_history) >= -1e-8))
  testthat::expect_identical(both$stop_rule, "both")
  if (both$converged) {
    testthat::expect_lte(both$kkt_violation_scaled, 1e-7)
  } else {
    testthat::expect_identical(both$convergence_reason, "max_iter")
  }
})

testthat::test_that("KKT and boundary controls validate inputs", {
  f <- .vcm_diag_fixture()
  testthat::expect_error(mrvcr::vcm(y ~ x, f$d, f$K,
    stop_rule = "maybe"), "stop_rule")
  testthat::expect_error(mrvcr::vcm(y ~ x, f$d, f$K,
    init = "unknown"), "init")
  testthat::expect_error(mrvcr::vcm(y ~ x, f$d, f$K,
    kkt_tol = 0), "kkt_tol")
  testthat::expect_error(mrvcr::vcm(y ~ x, f$d, f$K,
    boundary_tol = -1), "boundary_tol")
  testthat::expect_error(mrvcr::vcm(y ~ x, f$d, f$K,
    theta_init = c(0, .1)), "strictly positive")
})

# A score at theta ~ 0 must satisfy the one-sided condition s <= 0,
# not the unconstrained condition s == 0.
testthat::test_that("near-boundary estimates have finite diagnostic output", {
  set.seed(508)
  n <- 30L
  d <- data.frame(x = stats::rnorm(n))
  d$y <- 0.8 + 0.4 * d$x + stats::rnorm(n, sd = .4)
  Z <- stats::model.matrix(~ 0 + factor(rep(1:10, each = 3)), d)
  K <- list(group = tcrossprod(Z), residual = diag(n))
  fit <- suppressWarnings(mrvcr::vcm(y ~ x, d, K,
    pet = "REML", method = "MM", theta_init = c(.01, .2),
    max_iter = 100L, tol = 1e-9))
  testthat::expect_true(all(fit$variance_components >= 0))
  testthat::expect_true(is.finite(fit$kkt_violation_scaled))
  testthat::expect_true(all(is.finite(fit$score)))
  testthat::expect_true(all(diff(fit$loglik_history) >= -1e-7))
})
