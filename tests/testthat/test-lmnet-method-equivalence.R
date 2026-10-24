# Verify that the PET (statistical objective) is independent of the method.
# The existing Rust solver already supports Ridge + CoordinateDescent as well
# as ElasticNet(alpha=0) and ElasticNet(alpha=1).

.make_method_equivalence_data <- function() {
  set.seed(20261024)
  n <- 180L
  x1 <- stats::rnorm(n)
  x2 <- 0.7 * x1 + sqrt(1 - 0.7^2) * stats::rnorm(n)
  x3 <- stats::rnorm(n)
  d <- data.frame(x1 = x1, x2 = x2, x3 = x3)
  d$y <- 1.8 + 1.1*x1 - 0.8*x2 + 0.5*x3 + stats::rnorm(n, sd = 0.6)
  d
}

.test_mrvcr_fit <- function(d, pet, lambda, method = "auto", alpha = NULL,
                            ...) {
  mrvcr::lmnet(y ~ x1 + x2 + x3, data = d, pet = pet,
               lambda = lambda, method = method, alpha = alpha,
               tol = 1e-10, max_iter = 20000L, ...)
}

testthat::test_that("Ridge coordinate descent equals Cholesky, QR, and exact ridge", {
  d <- .make_method_equivalence_data()
  for (lambda in c(0.04, 0.2, 0.75)) {
    closed <- .test_mrvcr_fit(d, "Ridge", lambda, "cholesky")
    qr <- .test_mrvcr_fit(d, "Ridge", lambda, "qr")
    cd <- .test_mrvcr_fit(d, "Ridge", lambda, "coordinate_descent")
    alias <- .test_mrvcr_fit(d, "Ridge", lambda, "cd")
    auto <- .test_mrvcr_fit(d, "Ridge", lambda, "auto")

    X <- stats::model.matrix(y ~ x1 + x2 + x3, data = d)
    S <- nrow(X)
    penalty <- diag(c(0, rep(1, ncol(X) - 1L)), ncol(X))
    exact <- as.vector(solve(crossprod(X) / S + lambda*penalty,
                             crossprod(X, d$y) / S))

    testthat::expect_identical(auto$method, "cholesky")
    testthat::expect_identical(cd$method, "coordinate_descent")
    testthat::expect_identical(alias$method, "coordinate_descent")
    testthat::expect_true(cd$converged)
    testthat::expect_true(cd$iterations >= 1L)
    testthat::expect_true(cd$kkt_violation < 1e-7)
    testthat::expect_equal(as.vector(stats::coef(cd)), exact, tolerance = 2e-7)
    testthat::expect_equal(as.vector(stats::coef(qr)), exact, tolerance = 2e-7)
    testthat::expect_equal(as.vector(stats::coef(closed)), exact, tolerance = 2e-7)
    testthat::expect_equal(as.vector(stats::coef(alias)), exact, tolerance = 2e-7)
    testthat::expect_equal(as.vector(stats::coef(auto)), exact, tolerance = 2e-7)
    testthat::expect_equal(cd$objective, closed$objective, tolerance = 1e-8)
  }
})

testthat::test_that("ElasticNet endpoints equal corresponding Ridge and Lasso PETs", {
  d <- .make_method_equivalence_data()
  lambda <- 0.15
  ridge_cd <- .test_mrvcr_fit(d, "Ridge", lambda, "coordinate_descent")
  enet_zero <- .test_mrvcr_fit(d, "ElasticNet", lambda, "coordinate_descent", alpha = 0)
  lasso_cd <- .test_mrvcr_fit(d, "Lasso", lambda, "coordinate_descent")
  enet_one <- .test_mrvcr_fit(d, "ElasticNet", lambda, "coordinate_descent", alpha = 1)
  enet_middle_auto <- .test_mrvcr_fit(d, "ElasticNet", lambda, "auto", alpha = 0.4)
  enet_middle_explicit <- .test_mrvcr_fit(d, "ElasticNet", lambda,
                                           "coordinate_descent", alpha = 0.4)

  for (fit in list(ridge_cd, enet_zero, lasso_cd, enet_one,
                   enet_middle_auto, enet_middle_explicit)) {
    testthat::expect_true(fit$converged)
    testthat::expect_identical(fit$method, "coordinate_descent")
  }
  testthat::expect_equal(as.vector(stats::coef(enet_zero)),
                         as.vector(stats::coef(ridge_cd)), tolerance = 2e-7)
  testthat::expect_equal(as.vector(stats::coef(enet_one)),
                         as.vector(stats::coef(lasso_cd)), tolerance = 2e-7)
  testthat::expect_equal(as.vector(stats::coef(enet_middle_auto)),
                         as.vector(stats::coef(enet_middle_explicit)), tolerance = 2e-7)
  testthat::expect_equal(enet_zero$objective, ridge_cd$objective, tolerance = 1e-8)
  testthat::expect_equal(enet_one$objective, lasso_cd$objective, tolerance = 1e-8)
})

testthat::test_that("Weighted multiresponse Ridge coordinate descent matches Cholesky", {
  d <- .make_method_equivalence_data()
  d$y2 <- -0.6 + 0.3*d$x1 + 0.7*d$x3 + stats::rnorm(nrow(d), sd = 0.4)
  d$w <- rep(c(0, 0.5, 1, 2, 3), length.out = nrow(d))
  d$off <- 0.4 + 0.15*d$x2
  args <- list(formula = cbind(y, y2) ~ x1 + x2 + x3, data = d,
               pet = "Ridge", lambda = 0.15, weights = d$w,
               offset = d$off, penalty.factor = c(0, 1.1, 0.6, 1.8),
               tol = 1e-10, max_iter = 20000L)
  closed <- do.call(mrvcr::lmnet, c(args, list(method = "cholesky")))
  cd <- do.call(mrvcr::lmnet, c(args, list(method = "coordinate_descent")))

  testthat::expect_true(cd$converged)
  testthat::expect_equal(unname(stats::coef(cd)), unname(stats::coef(closed)),
                         tolerance = 2e-6)
  testthat::expect_equal(unname(stats::fitted(cd)), unname(stats::fitted(closed)),
                         tolerance = 2e-6)
  testthat::expect_equal(cd$objective, closed$objective, tolerance = 1e-8)
})
