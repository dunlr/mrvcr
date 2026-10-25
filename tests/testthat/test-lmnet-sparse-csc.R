# These tests compare native CSC results against independent dense fits and
# closed-form ridge on the same unstandardized, weighted Gaussian objective.
testthat::skip_if_not_installed("Matrix")

.make_sparse_example <- function(seed = 471L, n = 100L, p = 7L) {
  set.seed(seed)
  a <- matrix(stats::rnorm(n * p), n, p)
  a[stats::runif(n * p) < 0.78] <- 0
  # ensure both empty entries and nontrivial columns, including a zero column
  a[, p] <- 0
  colnames(a) <- paste0("v", seq_len(p))
  xs <- methods::as(Matrix::Matrix(a, sparse = TRUE), "dgCMatrix")
  list(x = xs, a = a)
}

.test_sparse_dense <- function(pet, alpha = NULL, intercept = TRUE,
                               weighted = FALSE, multi = FALSE) {
  obj <- .make_sparse_example()
  x <- obj$x
  a <- obj$a
  n <- nrow(a)
  y <- drop(0.9 + a %*% c(1.2, -.4, rep.int(0, ncol(a)-2L)) +
              stats::rnorm(n, sd = 0.15))
  response <- if (multi) cbind(Y1 = y, Y2 = -0.3 + 0.7*y) else y
  off <- if (weighted) seq(-.1, .2, length.out = n) else NULL
  w <- if (weighted) seq(0.25, 2, length.out = n) else NULL
  pf <- c(1, 1.5, 0.6, 0, 1, 2, 1)
  lambda <- .23
  got <- mrvcr::lmnet_sparse(
    x, response, pet = pet, lambda = lambda, alpha = alpha,
    intercept = intercept, weights = w, offset = off,
    penalty.factor = pf, tol = 1e-10, max_iter = 20000L
  )
  testthat::expect_true(got$converged)
  testthat::expect_lt(got$kkt_violation, 1e-7)
  d <- as.data.frame(a)
  d$y1 <- y
  if (multi) d$y2 <- -0.3 + 0.7*y
  if (weighted) { d$w <- w; d$off <- off }
  f <- stats::reformulate(colnames(a), response = if (multi) "cbind(y1,y2)" else "y1",
                           intercept = intercept)
  # Evaluate explicit arguments in this environment, not through ... wrappers.
  ref <- mrvcr::lmnet(
    f, data = d, pet = pet, lambda = lambda, alpha = alpha,
    weights = w, offset = off,
    penalty.factor = if (intercept) c(0, pf) else pf,
    method = "coordinate_descent", tol = 1e-10, max_iter = 20000L
  )
  testthat::expect_equal(unname(stats::coef(got)), unname(stats::coef(ref)),
                         tolerance = 2e-6)
  testthat::expect_equal(unname(stats::fitted(got)), unname(stats::fitted(ref)),
                         tolerance = 2e-6)
  testthat::expect_equal(unname(stats::residuals(got)),
                         unname(stats::residuals(ref)), tolerance = 2e-6)
  testthat::expect_equal(as.numeric(got$objective), as.numeric(ref$objective),
                         tolerance = 2e-7)
  testthat::expect_equal(unname(stats::predict(got, newx = x, offset = off)),
                         unname(stats::fitted(got)), tolerance = 2e-6)
  if (weighted) {
    testthat::expect_error(stats::predict(got, newx = x), "offset")
  }
  invisible(got)
}

testthat::test_that("native sparse ridge matches dense ridge and KKT", {
  .test_sparse_dense("Ridge")
  .test_sparse_dense("Ridge", weighted = TRUE)
  .test_sparse_dense("Ridge", intercept = FALSE)
})

testthat::test_that("native sparse lasso matches dense coordinate descent", {
  .test_sparse_dense("Lasso")
  .test_sparse_dense("Lasso", weighted = TRUE, multi = TRUE)
})

testthat::test_that("native sparse elastic net matches dense with weighted offsets", {
  .test_sparse_dense("ElasticNet", alpha = .45)
  .test_sparse_dense("ElasticNet", alpha = .45, weighted = TRUE, multi = TRUE)
})

testthat::test_that("sparse ridge agrees with analytic weighted normal equations", {
  ex <- .make_sparse_example(seed = 99L, n = 110L, p = 7L)
  n <- nrow(ex$a)
  y <- 1.4 + drop(ex$a %*% c(.6, -.8, rep(0, 5))) + stats::rnorm(n, sd = .2)
  w <- seq(.5, 1.5, length.out = n)
  off <- seq(0, .3, length.out = n)
  lam <- .4
  pf <- c(1, 2, .3, 1.2, 1, .5, 1)
  fit <- mrvcr::lmnet_sparse(ex$x, y, pet = "Ridge", lambda = lam,
                             weights = w, offset = off, penalty.factor = pf,
                             tol = 1e-11)
  z <- cbind(1, ex$a)
  expected <- solve(crossprod(z, w*z) / sum(w) + diag(lam*c(0, pf)),
                    crossprod(z, w*(y-off)) / sum(w))
  testthat::expect_equal(unname(stats::coef(fit)), unname(as.vector(expected)),
                         tolerance = 1e-7)
})

testthat::test_that("sparse input validation rejects unsafe and unsupported values", {
  ex <- .make_sparse_example()
  y <- stats::rnorm(nrow(ex$x))
  testthat::expect_error(mrvcr::lmnet_sparse(ex$a, y, lambda = 1), "dgCMatrix")
  testthat::expect_error(mrvcr::lmnet_sparse(ex$x, y, lambda = c(1, .5)), "one")
  testthat::expect_error(mrvcr::lmnet_sparse(ex$x, y[-1], lambda = 1), "rows")
  testthat::expect_error(mrvcr::lmnet_sparse(ex$x, y, lambda = 1,
                                              method = "cholesky"), "coordinate_descent")
  testthat::expect_error(mrvcr::lmnet_sparse(ex$x, y, lambda = 1,
                                              weights = rep(0, length(y))), "weights")
  testthat::expect_error(mrvcr::lmnet_sparse(ex$x, y, lambda = 1,
                                              penalty.factor = c(1,2)), "penalty.factor")
  testthat::expect_error(mrvcr::lmnet_sparse(ex$x, y, pet = "ElasticNet",
                                              lambda = 1), "alpha")
  damaged <- ex$x
  damaged@x[1] <- Inf
  testthat::expect_error(mrvcr::lmnet_sparse(damaged, y, lambda = 1), "Non-finite")
})
