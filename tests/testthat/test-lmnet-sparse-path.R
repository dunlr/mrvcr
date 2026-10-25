# Native sparse CSC warm-start paths, compared with scalar sparse fits.
testthat::skip_if_not_installed("Matrix")

.sparse_path_example <- function(n = 125L, p = 12L, seed = 889L) {
  set.seed(seed)
  a <- matrix(stats::rnorm(n * p), n, p)
  a[stats::runif(length(a)) < .84] <- 0
  a[, p] <- 0  # zero column must not cause numerical issues
  colnames(a) <- paste0("v", seq_len(p))
  x <- methods::as(Matrix::Matrix(a, sparse = TRUE), "dgCMatrix")
  y <- 1.4 + drop(a %*% c(.9, -.7, rep(0, p - 2L))) +
    stats::rnorm(n, sd = .2)
  list(x = x, a = a, y = y)
}

.test_sparse_path_scalar_parity <- function(pet, alpha = NULL,
                                            weighted = FALSE,
                                            multi = FALSE,
                                            intercept = TRUE) {
  ex <- .sparse_path_example()
  n <- nrow(ex$x)
  y <- if (multi) cbind(R1 = ex$y, R2 = -.2 + 1.3 * ex$y) else ex$y
  w <- if (weighted) seq(.5, 2, length.out = n) else NULL
  off <- if (weighted) seq(-.2, .1, length.out = n) else NULL
  pf <- rep(1.0, ncol(ex$x))
  pf[2] <- 1.8
  pf[3] <- 0.4
  pf[4] <- 0.0
  lambda <- c(.8, .4, .18, .05)
  path <- mrvcr::lmnet_sparse(
    ex$x, y, pet = pet, lambda = lambda, alpha = alpha,
    weights = w, offset = off, penalty.factor = pf,
    intercept = intercept, keep_x = TRUE, tol = 1e-10, max_iter = 20000L
  )
  testthat::expect_s3_class(path, "mrvcr_lmnet_sparse_path")
  testthat::expect_length(path$objective, length(lambda))
  testthat::expect_length(path$iterations, length(lambda))
  testthat::expect_length(path$converged, length(lambda))
  testthat::expect_true(all(path$converged))
  testthat::expect_lt(max(path$kkt_violation), 1e-7)
  for (lam in lambda) {
    ref <- mrvcr::lmnet_sparse(
      ex$x, y, pet = pet, lambda = lam, alpha = alpha,
      weights = w, offset = off, penalty.factor = pf,
      intercept = intercept, tol = 1e-10, max_iter = 20000L
    )
    testthat::expect_equal(unname(stats::coef(path, s = lam)),
                           unname(stats::coef(ref)), tolerance = 1e-6)
    testthat::expect_equal(as.numeric(path$objective[match(lam, lambda)]),
                           as.numeric(ref$objective), tolerance = 1e-7)
    testthat::expect_equal(unname(stats::predict(path, s = lam)),
                           unname(stats::fitted(ref)), tolerance = 1e-6)
    testthat::expect_equal(unname(stats::predict(path, newx = ex$x,
                                                 offset = off, s = lam)),
                           unname(stats::fitted(ref)), tolerance = 1e-6)
  }
  pred <- stats::predict(path, newx = ex$x, offset = off)
  if (multi) {
    testthat::expect_identical(dim(stats::coef(path)),
                               c(as.integer(ncol(ex$x) + intercept), 2L, length(lambda)))
    testthat::expect_identical(dim(pred), c(as.integer(n), 2L, length(lambda)))
    testthat::expect_identical(dim(stats::residuals(path)), dim(pred))
  } else {
    testthat::expect_identical(dim(stats::coef(path)),
                               c(as.integer(ncol(ex$x) + intercept), length(lambda)))
    testthat::expect_identical(dim(pred), c(as.integer(n), length(lambda)))
    testthat::expect_equal(unname(stats::fitted(path)), unname(pred), tolerance = 1e-10)
    testthat::expect_equal(unname(stats::residuals(path)),
                           unname(sweep(-pred, 1L, ex$y, `+`)), tolerance = 1e-8)
  }
  invisible(path)
}

testthat::test_that("CSC lasso path warm starts match independent sparse fits", {
  .test_sparse_path_scalar_parity("Lasso")
})

testthat::test_that("CSC elastic net weighted offset paths match scalar fits", {
  .test_sparse_path_scalar_parity("ElasticNet", alpha = .45, weighted = TRUE)
  .test_sparse_path_scalar_parity("ElasticNet", alpha = .45, weighted = TRUE,
                                  multi = TRUE)
})

testthat::test_that("CSC ridge paths match scalar fits with/without intercept", {
  .test_sparse_path_scalar_parity("Ridge")
  .test_sparse_path_scalar_parity("Ridge", intercept = FALSE)
})

testthat::test_that("sparse path does not silently retain design without keep_x", {
  ex <- .sparse_path_example()
  lambda <- c(.6, .2, .08)
  fit <- mrvcr::lmnet_sparse(ex$x, ex$y, pet = "Lasso", lambda = lambda,
                             keep_x = FALSE)
  testthat::expect_null(fit[["x", exact = TRUE]])
  testthat::expect_error(stats::predict(fit), "newx")
  testthat::expect_error(stats::fitted(fit), "newx")
  testthat::expect_error(stats::residuals(fit), "keep_x")
  testthat::expect_equal(dim(stats::predict(fit, newx = ex$x)),
                         c(as.integer(nrow(ex$x)), length(lambda)))
  testthat::expect_error(stats::coef(fit, s = .1234567), "exactly match")
  testthat::expect_error(stats::predict(fit, newx = ex$x[, -1]), "columns")
  testthat::expect_error(stats::predict(fit, newx = ex$x,
                                        s = .1234567), "exactly match")
  testthat::expect_identical(fit$method, "coordinate_descent")
})

testthat::test_that("sparse path validates lambda, PET, and method choices", {
  ex <- .sparse_path_example()
  f <- function(lambda, ...) mrvcr::lmnet_sparse(ex$x, ex$y, lambda = lambda, ...)
  testthat::expect_error(f(c(.2, .5)), "strictly decreasing")
  testthat::expect_error(f(c(.3, .3)), "strictly decreasing")
  testthat::expect_error(f(c(1, NA_real_)), "finite")
  testthat::expect_error(f(c(1, -.2)), "nonnegative")
  testthat::expect_error(f(c(1, .2), method = "cholesky"), "coordinate_descent")
  testthat::expect_error(f(c(1, .2), pet = "ElasticNet"), "alpha")
  testthat::expect_error(f(c(1, .2), pet = "NotAPet"), "PET|pet|unknown|Unsupported")
  # Scalars still use the previously validated sparse fit type.
  scalar <- f(.3, pet = "Ridge")
  testthat::expect_s3_class(scalar, "mrvcr_lmnet_sparse")
})

testthat::test_that("sparse paths work on all-zero design and no intercept", {
  x <- methods::as(Matrix::Matrix(matrix(0, 12, 3), sparse = TRUE), "dgCMatrix")
  colnames(x) <- c("a", "b", "c")
  y <- rep.int(1.0, 12L)
  fit <- mrvcr::lmnet_sparse(x, y, pet = "Lasso", lambda = c(.8, .2),
                             intercept = FALSE, keep_x = TRUE)
  testthat::expect_true(all(fit$converged))
  testthat::expect_equal(unname(stats::coef(fit)), matrix(0, 3L, 2L))
  testthat::expect_equal(unname(stats::predict(fit)), matrix(0, 12L, 2L))
})
