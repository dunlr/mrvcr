# Regularization path tests compare against independently fitted single-lambda
# problems. Single-lambda fits always use the original working implementation.
.make_path_data <- function() {
  set.seed(271828)
  n <- 120L
  x1 <- stats::rnorm(n)
  x2 <- 0.28 * x1 + stats::rnorm(n)
  x3 <- stats::rnorm(n)
  y <- 1.8 + 1.2*x1 - .7*x2 + .2*x3 + stats::rnorm(n, sd = .25)
  data.frame(y = y, x1 = x1, x2 = x2, x3 = x3,
             b = -0.4 + .6*x1 + .9*x3 + stats::rnorm(n, sd=.4),
             w = stats::runif(n,.3,2), off = stats::runif(n,-.15,.15))
}

.test_path_against_single <- function(pet, alpha = NULL) {
  d <- .make_path_data()
  ls <- c(.8, .3, .1, .02)
  path <- mrvcr::lmnet(y ~ x1+x2+x3, data = d, pet = pet,
                       lambda = ls, alpha = alpha, tol = 1e-10)
  testthat::expect_s3_class(path, "mrvcr_lmnet_path")
  testthat::expect_equal(path$lambda, ls)
  testthat::expect_identical(path$method, "coordinate_descent")
  testthat::expect_true(all(path$converged))
  testthat::expect_equal(dim(stats::coef(path)), c(4L, length(ls)))
  testthat::expect_equal(dim(stats::predict(path)), c(nrow(d), length(ls)))
  testthat::expect_null(path$fitted.values)
  for (k in seq_along(ls)) {
    single <- mrvcr::lmnet(y ~ x1+x2+x3, data = d, pet = pet,
                           lambda = ls[k], alpha = alpha,
                           method = "coordinate_descent", tol = 1e-10)
    testthat::expect_equal(unname(stats::coef(path, s = ls[k])),
                           unname(stats::coef(single)), tolerance = 2e-6)
    testthat::expect_equal(unname(stats::predict(path, s = ls[k])),
                           unname(stats::predict(single)), tolerance = 2e-6)
    testthat::expect_equal(path$objective[k], single$objective,
                           tolerance = 1e-7)
    testthat::expect_equal(path$weighted.rss[k], single$weighted.rss,
                           tolerance = 1e-7)
  }
}

testthat::test_that("warm start lasso path reproduces scalar solutions", {
  .test_path_against_single("Lasso")
})

testthat::test_that("warm start elastic net reproduces scalar solutions", {
  .test_path_against_single("ElasticNet", alpha = 0.45)
})

testthat::test_that("warm start ridge path reproduces scalar and analytic solutions", {
  .test_path_against_single("Ridge")
  d <- .make_path_data()
  ls <- c(1, .3, .07)
  path <- mrvcr::lmnet(y ~ x1 + x2 + x3, data=d, pet="Ridge", lambda=ls)
  for (k in seq_along(ls)) {
    b_ref <- as.vector(solve(crossprod(cbind(1, as.matrix(d[,c("x1","x2","x3")])))/nrow(d) +
       ls[k] * diag(c(0,1,1,1)),
       crossprod(cbind(1, as.matrix(d[,c("x1","x2","x3")])), d$y)/nrow(d)))
    testthat::expect_equal(unname(stats::coef(path, s=ls[k])), b_ref,
                           tolerance=2e-6)
  }
})

testthat::test_that("weighted offsets, nonuniform penalties, and newdata predictions work", {
  d <- .make_path_data()
  ls <- c(.6,.15,.02)
  pf <- c(0,.7, 1.2, .5)
  d$internal_weight <- d$w
  d$internal_offset <- d$off
  path <- mrvcr::lmnet(y ~ x1+x2+x3, data = d, weights = internal_weight,
                       offset = internal_offset, penalty.factor=pf,
                       pet="ElasticNet", alpha=.5, lambda=ls, tol=1e-10)
  testthat::expect_true(all(path$converged))
  for (lam in ls) {
    single <- mrvcr::lmnet(y ~ x1+x2+x3, data=d, weights=internal_weight,
                           offset=internal_offset, penalty.factor=pf,
                           pet="ElasticNet", alpha=.5, lambda=lam, tol=1e-10)
    testthat::expect_equal(unname(stats::coef(path,s=lam)),
                           unname(stats::coef(single)), tolerance=2e-6)
    testthat::expect_equal(unname(stats::predict(path,s=lam)),
                           unname(stats::fitted(single)), tolerance=2e-6)
  }
  testthat::expect_error(stats::predict(path, newdata = d[1:5,]), "offset")
  manual <- stats::model.matrix(~ x1+x2+x3,d[1:5,]) %*%
            stats::coef(path, s=ls[2]) + d$off[1:5]
  testthat::expect_equal(unname(stats::predict(path,newdata=d[1:5,],s=ls[2],
                                                  offset=d$off[1:5])),
                         unname(as.vector(manual)), tolerance=1e-7)
})

testthat::test_that("multivariate paths retain coefficient and prediction dimensions", {
  d <- .make_path_data()
  ls <- c(.6,.2,.05)
  path <- mrvcr::lmnet(cbind(y,b) ~ x1+x2+x3, data=d, pet="Lasso",
                       lambda=ls,tol=1e-10)
  testthat::expect_true(all(path$converged))
  testthat::expect_equal(dim(stats::coef(path)), c(4L,2L,3L))
  testthat::expect_equal(dim(stats::predict(path)), c(nrow(d),2L,3L))
  testthat::expect_equal(dim(stats::coef(path,s=ls[1])), c(4L,2L))
  testthat::expect_equal(dim(stats::predict(path,s=ls[1])), c(nrow(d),2L))
  testthat::expect_equal(dim(stats::residuals(path)), c(nrow(d),2L,3L))
  for (k in seq_along(ls)) {
    single <- mrvcr::lmnet(cbind(y,b) ~ x1+x2+x3, data=d, pet="Lasso",
                           lambda=ls[k],tol=1e-10)
    testthat::expect_equal(unname(stats::coef(path,s=ls[k])),
                           unname(stats::coef(single)), tolerance=2e-6)
  }
})

testthat::test_that("path arguments and numerical-method dispatch fail informatively", {
  d <- .make_path_data()
  testthat::expect_error(mrvcr::lmnet(y~x1, data=d, lambda=c(.1,.2)), "decreasing")
  testthat::expect_error(mrvcr::lmnet(y~x1, data=d, lambda=c(.1,.1)), "decreasing")
  testthat::expect_error(mrvcr::lmnet(y~x1, data=d, lambda=c(.1,NaN)), "finite")
  testthat::expect_error(mrvcr::lmnet(y~x1, data=d, lambda=c(.1,-.1)), "nonnegative")
  testthat::expect_error(mrvcr::lmnet(y~x1, data=d, lambda=c(.5,.1),
                                     method="cholesky"), "coordinate_descent|paths support")
  fit <- mrvcr::lmnet(y~x1, data=d, pet="Lasso", lambda=c(.5,.1))
  testthat::expect_error(stats::coef(fit,s=.2), "exactly match")
  testthat::expect_error(stats::predict(fit,newdata=d[1:3,],s=.2), "exactly match")
  testthat::expect_equal(dim(stats::coef(fit,s=c(.1,.5))),c(2L,2L))
})
