# Spectral two-kernel VCM must optimize the *same* Gaussian ML/REML
# objective and use the same EM/MM mappings as the dense solver.
.make_spectral_example <- function(seed = 741L) {
  set.seed(seed)
  n <- 36L
  d <- data.frame(x = rnorm(n), group = factor(rep(seq_len(9L), each = 4L)))
  d$y <- 0.8 + 0.55 * d$x + rnorm(9L, sd = 0.85)[as.integer(d$group)] +
    rnorm(n, sd = 0.35)
  Z <- stats::model.matrix(~ 0 + group, data = d)
  list(data = d, K = list(random = tcrossprod(Z), residual = diag(n)))
}

.fit_spectral_pair <- function(d, K, pet, method, max_iter = 5L,
                               stop_rule = "both") {
  args <- list(formula = y ~ x, data = d, kernels = K, pet = pet,
               method = method, theta_init = c(0.75, 0.25),
               max_iter = max_iter, tol = 1e-13,
               kkt_tol = 1e-12, stop_rule = stop_rule)
  dense <- suppressWarnings(do.call(mrvcr::vcm, c(args, list(solver = "dense"))))
  spectral <- suppressWarnings(do.call(mrvcr::vcm, c(args, list(solver = "spectral"))))
  list(dense = dense, spectral = spectral)
}

testthat::test_that("spectral ML and REML reproduce dense EM/MM trajectories", {
  data <- .make_spectral_example()
  for (pet in c("MLE", "REML")) for (method in c("EM", "MM")) {
    fits <- .fit_spectral_pair(data$data, data$K, pet, method)
    a <- fits$dense; b <- fits$spectral
    testthat::expect_equal(b$solver, "spectral")
    testthat::expect_equal(a$solver, "dense")
    testthat::expect_equal(b$theta_initial, a$theta_initial)
    testthat::expect_equal(b$variance_components, a$variance_components,
                           tolerance = 2e-7)
    testthat::expect_equal(b$log_likelihood, a$log_likelihood,
                           tolerance = 2e-7)
    testthat::expect_equal(unname(b$loglik_history), unname(a$loglik_history),
                           tolerance = 2e-7)
    testthat::expect_equal(b$score, a$score, tolerance = 2e-6)
    testthat::expect_equal(b$kkt_violation_scaled, a$kkt_violation_scaled,
                           tolerance = 2e-7)
    testthat::expect_equal(unname(stats::coef(b)), unname(stats::coef(a)),
                           tolerance = 2e-7)
    testthat::expect_equal(unname(stats::fitted(b)), unname(stats::fitted(a)),
                           tolerance = 2e-7)
    testthat::expect_equal(unname(stats::residuals(b)), unname(stats::residuals(a)),
                           tolerance = 2e-7)
    testthat::expect_true(all(diff(b$loglik_history) >= -1e-7))
  }
})

testthat::test_that("spectral and dense solutions agree at convergence", {
  data <- .make_spectral_example(777L)
  for (pet in c("MLE", "REML")) for (method in c("EM", "MM")) {
    args <- list(y ~ x, data$data, data$K, pet = pet, method = method,
                 theta_init = c(0.6, 0.3), max_iter = 800L,
                 tol = 1e-7, kkt_tol = 1e-5, stop_rule = "both")
    dense <- suppressWarnings(do.call(mrvcr::vcm,
                                      c(args, list(solver = "dense"))))
    spectral <- suppressWarnings(do.call(mrvcr::vcm,
                                         c(args, list(solver = "spectral"))))
    testthat::expect_true(dense$converged)
    testthat::expect_true(spectral$converged)
    testthat::expect_equal(spectral$variance_components,
                           dense$variance_components, tolerance = 3e-5)
    testthat::expect_equal(spectral$log_likelihood, dense$log_likelihood,
                           tolerance = 3e-6)
    testthat::expect_true(spectral$kkt_violation_scaled <= spectral$kkt_tol)
  }
})

testthat::test_that("nonidentity positive definite reference kernel is supported", {
  data <- .make_spectral_example(77L)
  n <- nrow(data$data)
  # A nontrivial, dense, positive-definite second kernel.
  t <- seq_len(n)
  ref <- outer(t, t, function(a, b) 0.42 ^ abs(a - b))
  K <- list(random = data$K$random, correlated_residual = ref)
  fit <- .fit_spectral_pair(data$data, K, "REML", "MM", max_iter = 8L)
  testthat::expect_equal(fit$spectral$loglik_history,
                         fit$dense$loglik_history, tolerance = 2e-7)
  testthat::expect_equal(fit$spectral$variance_components,
                         fit$dense$variance_components, tolerance = 2e-7)
})

testthat::test_that("two full-rank covariance kernels work with both PETs", {
  data <- .make_spectral_example(202L)
  n <- nrow(data$data)
  K <- list(kinship = 0.07 * data$K$random + diag(n), residual = diag(n))
  for (pet in c("MLE", "REML")) {
    fit <- .fit_spectral_pair(data$data, K, pet, "EM", max_iter = 6L)
    testthat::expect_equal(fit$spectral$loglik_history,
                           fit$dense$loglik_history, tolerance = 2e-7)
  }
})

testthat::test_that("spectral solver rejects incompatible kernel structure", {
  data <- .make_spectral_example()
  n <- nrow(data$data)
  testthat::expect_error(mrvcr::vcm(y ~ x, data$data,
    kernels = list(random = data$K[[1L]], residual = diag(n), third = diag(n)),
    solver = "spectral"), "exactly two")
  testthat::expect_error(mrvcr::vcm(y ~ x, data$data,
    kernels = list(residual = diag(n), random = data$K[[1L]]),
    solver = "spectral"), "second covariance kernel must be positive definite")
  testthat::expect_error(mrvcr::vcm(y ~ x, data$data, kernels = data$K,
    solver = "invalid"), "solver")
})

testthat::test_that("spectral predictions retain original fixed effect scale", {
  data <- .make_spectral_example(83L)
  a <- mrvcr::vcm(y ~ x, data$data, data$K, solver = "spectral",
                  method = "MM", pet = "REML")
  expected <- as.vector(stats::model.matrix(~ x, data$data) %*% stats::coef(a))
  testthat::expect_equal(unname(stats::fitted(a)), expected,
                         tolerance = 1e-10)
  testthat::expect_equal(unname(stats::predict(a, newdata = data$data[1:4,])),
                         expected[1:4], tolerance = 1e-10)
})
