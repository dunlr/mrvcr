# SQUAREM is an accelerator of the existing exact Gaussian EM/MM fixed-point
# maps. Both dense and spectral implementations should match the unaccelerated
# likelihood maximum while recording all attempted objective evaluations.

.vcm_squarem_data <- function(seed = 905L, n_groups = 8L) {
  set.seed(seed)
  g <- factor(rep(seq_len(n_groups), each = 4L))
  n <- length(g)
  d <- data.frame(x = rnorm(n), group = g)
  d$y <- 1.1 + 0.4*d$x + rnorm(n_groups, sd = 0.8)[as.integer(g)] +
    rnorm(n, sd = 0.3)
  Z <- stats::model.matrix(~ 0 + group, data = d)
  list(d = d, kernels = list(random = tcrossprod(Z), residual = diag(n)))
}

.vcm_squarem_fit <- function(dat, pet, method, solver, acceleration,
                             max_iter = 250L) {
  suppressWarnings(mrvcr::vcm(y ~ x, data = dat$d,
    kernels = dat$kernels, pet = pet, method = method,
    solver = solver, acceleration = acceleration,
    theta_init = c(random = 0.6, residual = 0.2),
    tol = 1e-6, kkt_tol = 1e-5, stop_rule = "both",
    max_iter = max_iter))
}

testthat::test_that("no acceleration preserves the legacy EM/MM mapping", {
  dat <- .vcm_squarem_data()
  for (solver in c("dense", "spectral")) for (method in c("EM", "MM")) {
    before <- suppressWarnings(mrvcr::vcm(y ~ x, dat$d, dat$kernels,
      method = method, solver = solver, theta_init = c(0.4, 0.25),
      max_iter = 4L, tol = 1e-13, kkt_tol = 1e-12, stop_rule = "both"))
    after <- suppressWarnings(mrvcr::vcm(y ~ x, dat$d, dat$kernels,
      method = method, solver = solver, acceleration = "none",
      theta_init = c(0.4, 0.25), max_iter = 4L,
      tol = 1e-13, kkt_tol = 1e-12, stop_rule = "both"))
    testthat::expect_identical(before$loglik_history, after$loglik_history)
    testthat::expect_identical(before$variance_components, after$variance_components)
    testthat::expect_equal(after$fixed_point_evaluations, after$iterations)
    testthat::expect_gte(after$likelihood_evaluations,
                         after$fixed_point_evaluations + 1L)
    testthat::expect_identical(after$squarem_attempts, 0L)
  }
})

testthat::test_that("SQUAREM uses the same ML/REML objective and remains monotone", {
  dat <- .vcm_squarem_data(271L)
  for (pet in c("MLE", "REML")) for (method in c("EM", "MM")) {
    for (solver in c("dense", "spectral")) {
      ref <- .vcm_squarem_fit(dat, pet, method, solver, "none")
      sq <- .vcm_squarem_fit(dat, pet, method, solver, "SQUAREM")
      testthat::expect_true(ref$converged)
      testthat::expect_true(sq$converged)
      testthat::expect_equal(sq$log_likelihood, ref$log_likelihood,
                             tolerance = 3e-5)
      testthat::expect_equal(sq$variance_components, ref$variance_components,
                             tolerance = 2e-3)
      testthat::expect_lte(sq$kkt_violation_scaled, 1e-5)
      testthat::expect_true(all(diff(sq$loglik_history) >= -1e-7))
      testthat::expect_equal(length(sq$loglik_history), sq$iterations + 1L)
      testthat::expect_true(sq$fixed_point_evaluations >= 2L * sq$iterations)
      testthat::expect_gte(sq$likelihood_evaluations,
                           sq$fixed_point_evaluations + 1L)
      testthat::expect_equal(sq$squarem_attempts,
                             sq$squarem_accepted + sq$squarem_rejected)
      testthat::expect_identical(sq$acceleration, "SQUAREM")
    }
  }
})

testthat::test_that("SQUAREM cycles, budgets and legacy wrapper work", {
  dat <- .vcm_squarem_data(602L)
  for (solver in c("dense", "spectral")) {
    sq <- .vcm_squarem_fit(dat, "REML", "EM", solver, "SQUAREM",
                          max_iter = 1L)
    testthat::expect_lte(sq$iterations, 1L)
    testthat::expect_equal(length(sq$loglik_history), sq$iterations + 1L)
    testthat::expect_true(all(is.finite(sq$variance_components)))
    testthat::expect_true(all(sq$variance_components >= 0))
    testthat::expect_gte(sq$fixed_point_evaluations, 2L * sq$iterations)
  }
  fit <- suppressWarnings(mrvcr::vcm_zzt(y ~ x, ~ 0 + group,
       data = dat$d, method = "MM", acceleration = "SQUAREM",
       max_iter = 20L))
  testthat::expect_identical(fit$acceleration, "SQUAREM")
  testthat::expect_gte(fit$likelihood_evaluations, 1L)
})

testthat::test_that("invalid acceleration is rejected at the R boundary", {
  dat <- .vcm_squarem_data()
  testthat::expect_error(mrvcr::vcm(y ~ x, dat$d, dat$kernels,
                            acceleration = "fast"), "acceleration")
  testthat::expect_error(mrvcr::vcm(y ~ x, dat$d, dat$kernels,
                            acceleration = NA_character_), "acceleration")
  testthat::expect_error(mrvcr::vcm(y ~ x, dat$d, dat$kernels,
                            acceleration = 2L), "acceleration")
})

testthat::test_that("dense SQUAREM supports more than two PSD kernels", {
  dat <- .vcm_squarem_data(733L, n_groups = 5L)
  n <- nrow(dat$d)
  j <- seq_len(n)
  extra <- outer(j, j, function(a, b) 0.3 ^ abs(a - b))
  kernels <- c(dat$kernels, list(correlated = extra))
  for (method in c("EM", "MM")) {
    fit <- suppressWarnings(mrvcr::vcm(y ~ x, dat$d, kernels,
      pet = "MLE", method = method, acceleration = "SQUAREM",
      theta_init = c(0.4, 0.2, 0.1), max_iter = 5L,
      stop_rule = "both", tol = 1e-11, kkt_tol = 1e-11))
    testthat::expect_equal(length(fit$variance_components), 3L)
    testthat::expect_true(all(diff(fit$loglik_history) >= -1e-7))
    testthat::expect_equal(length(fit$loglik_history), fit$iterations + 1L)
    testthat::expect_true(all(is.finite(fit$variance_components)))
  }
})
