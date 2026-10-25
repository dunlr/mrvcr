# Regression tests for optional `$x` versus `$xlevels` partial matching.
# A path keeps xlevels for factors even when x = FALSE, so object$x is NOT a
# reliable way to detect whether a design matrix was retained.

.make_prediction_path_data <- function() {
  set.seed(20261024)
  n <- 60L
  z <- stats::rnorm(n)
  g <- factor(rep(c("A", "B", "C"), length.out = n))
  off <- stats::runif(n, -.15, .15)
  w <- stats::runif(n, .5, 2)
  y <- 1.4 + .85 * z + ifelse(g == "C", -.4, .3) + off +
    stats::rnorm(n, sd = .16)
  z2 <- -.8 + .25 * z + ifelse(g == "B", .6, -.2) +
    stats::rnorm(n, sd = .18)
  data.frame(y = y, z2 = z2, z = z, g = g, off = off, w = w)
}

testthat::test_that("path predicts from saved model when optional x is absent", {
  d <- .make_prediction_path_data()
  lambda <- c(.7, .2, .05)
  path <- mrvcr::lmnet(y ~ z + g, data = d, pet = "Lasso",
                       lambda = lambda, tol = 1e-10)
  testthat::expect_null(path[["x", exact = TRUE]])
  testthat::expect_true(is.list(path$xlevels))
  testthat::expect_true(length(path$xlevels) > 0L)
  testthat::expect_true(is.data.frame(path[["model", exact = TRUE]]))
  xmat <- stats::model.matrix(y ~ z + g, data = d)
  testthat::expect_equal(unname(stats::predict(path)),
                         unname(xmat %*% stats::coef(path)),
                         tolerance = 1e-7)
  testthat::expect_equal(unname(stats::fitted(path)),
                         unname(stats::predict(path)), tolerance = 1e-7)
  one <- stats::predict(path, s = lambda[2L])
  testthat::expect_equal(unname(one),
                         unname(drop(xmat %*% stats::coef(path, s = lambda[2L]))),
                         tolerance = 1e-7)
  testthat::expect_equal(unname(stats::predict(path, newdata = d[1:6, ], s = lambda[2L])),
                         unname(drop(xmat[1:6, , drop = FALSE] %*%
                                       stats::coef(path, s = lambda[2L]))),
                         tolerance = 1e-7)
})

testthat::test_that("path uses explicitly retained x and rejects absent training data", {
  d <- .make_prediction_path_data()
  lambda <- c(.6, .15)
  f <- y ~ z + g
  kept <- mrvcr::lmnet(f, data = d, pet = "Ridge", lambda = lambda,
                       x = TRUE, model = FALSE, tol = 1e-10)
  testthat::expect_true(is.matrix(kept[["x", exact = TRUE]]))
  testthat::expect_null(kept[["model", exact = TRUE]])
  testthat::expect_equal(unname(stats::predict(kept)),
                         unname(stats::model.matrix(f, d) %*% stats::coef(kept)),
                         tolerance = 1e-7)

  stripped <- mrvcr::lmnet(f, data = d, pet = "Ridge", lambda = lambda,
                           x = FALSE, model = FALSE)
  testthat::expect_error(stats::predict(stripped), "model = TRUE or x = TRUE")
  # Explicit newdata needs neither a retained model frame nor design matrix.
  testthat::expect_equal(
    unname(stats::predict(stripped, newdata = d[1:5, ], s = lambda[1L])),
    unname(as.vector(stats::model.matrix(f, d[1:5, ]) %*%
                       stats::coef(stripped, s = lambda[1L]))),
    tolerance = 1e-7)
})

testthat::test_that("path predicts weighted offset and multivariate fits", {
  d <- .make_prediction_path_data()
  lambda <- c(.45, .13)
  f <- y ~ z + g
  path <- mrvcr::lmnet(f, data = d, pet = "ElasticNet", alpha = .4,
                       lambda = lambda, weights = w, offset = off, tol = 1e-10)
  xmat <- stats::model.matrix(f, data = d)
  wanted <- sweep(xmat %*% stats::coef(path), 1L, d$off, "+")
  testthat::expect_equal(unname(stats::predict(path)), unname(wanted),
                         tolerance = 1e-7)
  testthat::expect_error(stats::predict(path, newdata = d[1:4, ],
                                        s = lambda[1L]), "offset")
  testthat::expect_equal(
    unname(stats::predict(path, newdata = d[1:4, ], s = lambda[2L],
                          offset = d$off[1:4])),
    unname(as.vector(xmat[1:4, , drop = FALSE] %*%
                       stats::coef(path, s = lambda[2L])) + d$off[1:4]),
    tolerance = 1e-7)

  multi <- mrvcr::lmnet(cbind(y, z2) ~ z + g, data = d, pet = "Lasso",
                        lambda = lambda, tol = 1e-10)
  testthat::expect_equal(dim(stats::predict(multi)),
                         c(nrow(d), 2L, length(lambda)))
  mm <- stats::model.matrix(~ z + g, data = d)
  for (s in lambda) {
    testthat::expect_equal(
      unname(stats::predict(multi, s = s)),
      unname(mm %*% stats::coef(multi, s = s)), tolerance = 1e-7)
  }
  testthat::expect_equal(dim(stats::residuals(multi)),
                         c(nrow(d), 2L, length(lambda)))
})
