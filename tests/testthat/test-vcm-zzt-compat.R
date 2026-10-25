testthat::test_that("ZZT MM default is available and invalid designs are rejected", {
  testthat::expect_true(is.function(mrvcr::vcm_zzt))
  set.seed(159)
  d <- data.frame(x = stats::rnorm(12),
                  group = factor(rep(1:4, each = 3)))
  d$y <- 0.5 + d$x + rep(c(-1, 1, -.5, .5), each = 3) +
    stats::rnorm(12, sd = .25)
  default_fit <- suppressWarnings(mrvcr::vcm_zzt(
    y ~ x, ~ 0 + group, d, max_iter = 200L
  ))
  testthat::expect_equal(default_fit$method, "MM")
  testthat::expect_true(all(diff(default_fit$loglik_history) >= -1e-8))
  testthat::expect_error(mrvcr::vcm_zzt(y ~ x, y ~ group, d, method = "EM"),
                         "one-sided")
})

testthat::test_that("legacy ZZT EM adapter equals direct Gaussian VCM fit", {
  set.seed(157)
  n <- 24L
  d <- data.frame(x = stats::rnorm(n),
                  group = factor(rep(seq_len(6), each = 4)))
  d$y <- 0.8 + 1.2*d$x +
    stats::rnorm(6)[as.integer(d$group)] * 0.4 +
    stats::rnorm(n, sd = 0.5)
  Z <- stats::model.matrix(~ 0 + group, d)
  K <- list(random = tcrossprod(Z), residual = diag(n))

  for (pet in c("MLE", "REML")) for (algorithm in c("EM", "MM")) {
    args <- list(pet = pet, method = algorithm, theta_init = c(0.3, 0.4),
                 max_iter = 20L, tol = 1e-9)
    adapter <- suppressWarnings(do.call(mrvcr::vcm_zzt,
      c(list(formula = y ~ x, random_formula = ~ 0 + group, data = d), args)))
    direct <- suppressWarnings(do.call(mrvcr::vcm,
      c(list(formula = y ~ x, data = d, kernels = K), args)))

    testthat::expect_s3_class(adapter, "mrvcr_vcm_em")
    testthat::expect_equal(adapter$variance_components,
                           direct$variance_components, tolerance = 1e-10)
    testthat::expect_equal(adapter$log_likelihood,
                           direct$log_likelihood, tolerance = 1e-10)
    testthat::expect_equal(unname(stats::coef(adapter)),
                           unname(stats::coef(direct)), tolerance = 1e-10)
  }
})
