# Gaussian Ridge: objective is 1/(2*sum(w)) RSS + lambda/2 * beta' D beta.
# This analytic reference intentionally uses base R to independently verify
# Rust's faer solve, while remaining dependency-free and deterministic.
.ridge_reference <- function(X, Y, lambda, penalty.factor, weights = NULL, offset = NULL) {
  n <- nrow(X)
  p <- ncol(X)
  if (is.null(weights)) weights <- rep(1, n)
  if (is.null(offset)) offset <- rep(0, n)
  Y <- as.matrix(Y)
  s <- sum(weights)
  WX <- X * sqrt(weights / s)
  WY <- sweep(Y, 1L, offset, "-") * sqrt(weights / s)
  solve(crossprod(WX) + lambda * diag(penalty.factor, p), crossprod(WX, WY))
}

testthat::test_that("Gaussian Ridge matches analytic solution (both methods)", {
  set.seed(230)
  d <- data.frame(x = rnorm(120), z = rnorm(120))
  d$y <- 1.2 + 0.4 * d$x - 0.9 * d$z + rnorm(120, sd = 0.8)
  X <- stats::model.matrix(~ x + z, d)
  penalty <- c(0, 1, 1)
  expected <- .ridge_reference(X, d$y, lambda = 0.7, penalty.factor = penalty)
  for (method in c("cholesky", "qr", "householder_qr")) {
    fit <- mrvcr::lmnet(y ~ x + z, data = d, pet = "Ridge", lambda = 0.7,
                        method = method)
    testthat::expect_equal(unname(as.vector(coef(fit))),
                           as.vector(expected), tolerance = 1e-8)
    testthat::expect_equal(unname(as.vector(fitted(fit))),
                           as.vector(X %*% expected), tolerance = 1e-8)
    testthat::expect_equal(unname(as.vector(residuals(fit))),
                           as.vector(d$y - X %*% expected), tolerance = 1e-8)
    testthat::expect_equal(unname(as.vector(predict(fit, newdata = d[1:7,]))),
                           as.vector(X[1:7,] %*% expected), tolerance = 1e-8)
    testthat::expect_false(inherits(fit, "lm"))
  }
})

testthat::test_that("Weights, offsets, and penalty factors agree with weighted normal equations", {
  set.seed(231)
  d <- data.frame(x = rnorm(80), z = rnorm(80))
  d$w <- sample(c(0, 0.2, 1, 3), nrow(d), replace = TRUE)
  d$off <- 0.1 + 0.3 * d$z
  d$y <- 2 - 1.5 * d$x + 0.2 * d$z + d$off + rnorm(nrow(d))
  pf <- c(0, 2, 0.3)
  X <- stats::model.matrix(~ x + z, d)
  expected <- .ridge_reference(X, d$y, 0.2, pf, d$w, d$off)
  for (method in c("cholesky", "qr")) {
    fit <- mrvcr::lmnet(y ~ x + z, data = d, pet = "Ridge", lambda = 0.2,
                        weights = w, offset = off, penalty.factor = pf,
                        method = method)
    testthat::expect_equal(as.vector(coef(fit)), as.vector(expected), tolerance = 1e-8)
    testthat::expect_equal(as.vector(fitted(fit)),
                           as.vector(X %*% expected + d$off), tolerance = 1e-8)
    testthat::expect_error(predict(fit, newdata = d[1:4,]), "Provide 'offset'")
    testthat::expect_equal(as.vector(predict(fit, newdata = d[1:4,], offset = d$off[1:4])),
                           as.vector(X[1:4,] %*% expected + d$off[1:4]), tolerance = 1e-8)
    s <- sum(d$w)
    objective <- sum(d$w * residuals(fit)^2) / (2*s) +
      0.2/2 * sum(pf * as.vector(coef(fit))^2)
    testthat::expect_equal(fit$objective, objective, tolerance = 1e-9)
  }
})

testthat::test_that("Multiple responses reuse the same ridge system", {
  set.seed(232)
  d <- data.frame(x = rnorm(100), z = rnorm(100))
  d$a <- 1 + d$x + rnorm(100)
  d$b <- -2 + 0.2*d$x - 0.4*d$z + rnorm(100)
  X <- stats::model.matrix(~ x + z, d)
  Y <- as.matrix(d[c("a", "b")])
  expected <- .ridge_reference(X, Y, 1.3, c(0,1,1))
  fit <- mrvcr::lmnet(cbind(a,b) ~ x + z, data = d, lambda = 1.3)
  testthat::expect_equal(dim(coef(fit)), c(3L,2L))
  testthat::expect_equal(unname(coef(fit)), unname(expected), tolerance=1e-8)
  testthat::expect_equal(unname(fitted(fit)), unname(X %*% expected), tolerance=1e-8)
  testthat::expect_equal(unname(predict(fit, d[1:5,])),
                         unname(X[1:5,] %*% expected), tolerance=1e-8)
})

testthat::test_that("High dimensional ridge is identifiable with positive penalties", {
  set.seed(233)
  n <- 25L
  d <- as.data.frame(matrix(rnorm(n*65L), nrow=n))
  names(d) <- paste0("v", seq_len(65L))
  d$y <- rnorm(n)
  fit <- mrvcr::lmnet(y ~ ., data=d, lambda=2, method="cholesky")
  testthat::expect_equal(length(coef(fit)), 66L)
  testthat::expect_true(all(is.finite(coef(fit))))
  testthat::expect_true(is.finite(fit$objective))
})

testthat::test_that("Ridge lambda zero agrees with OLS on full rank X", {
  set.seed(234)
  d <- data.frame(x = rnorm(70), y = rnorm(70))
  fit <- mrvcr::lmnet(y ~ x, data=d, lambda=0)
  ref <- stats::lm(y ~ x, data=d)
  testthat::expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance=1e-9)
})

testthat::test_that("Bad hyperparameters and unavailable PETs error in Rust", {
  d <- data.frame(x = 1:10, y = 1:10)
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, lambda=-1), "lambda")
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, lambda=Inf), "lambda")
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, lambda=1e308,
                                   penalty.factor=c(0,1e308)), "overflows")
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, lambda=1, alpha=0.5), "alpha")
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, pet="Lasso", lambda=1), "not implemented")
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, pet="ElasticNet", lambda=1,
                                   alpha=0.5), "not implemented")
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, pet="NoSuchPet", lambda=1), "Unsupported")
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, lambda=1,
                                   penalty.factor=c(0,1,1)), "model-matrix column")
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, lambda=1,
                                   penalty.factor=c(0,-1)), "penalty.factor")
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, lambda=1,
                                   method="svd"), "Ridge supports")
})

testthat::test_that("Scaling prior weights does not change the fitted ridge coefficients", {
  set.seed(235)
  d <- data.frame(x=rnorm(40), y=rnorm(40), w=runif(40,0.1,2))
  a <- mrvcr::lmnet(y ~ x, data=d, weights=w, lambda=0.3)
  d$w2 <- 7*d$w
  b <- mrvcr::lmnet(y ~ x, data=d, weights=w2, lambda=0.3)
  testthat::expect_equal(unname(coef(a)), unname(coef(b)), tolerance=1e-8)
})
