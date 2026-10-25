# Independent Gaussian elastic-net regression tests.  Uses an orthogonal-design
# exact solution and KKT checks, rather than treating the Rust solver as truth.

.soft_threshold_ref <- function(x, t) sign(x) * pmax(abs(x) - t, 0)

.make_orthogonal_enet_data <- function() {
  set.seed(841)
  n <- 80L
  qq <- qr.Q(qr(cbind(rep(1, n), rnorm(n), rnorm(n))))
  x1 <- qq[, 2L] * sqrt(n)
  x2 <- qq[, 3L] * sqrt(n)
  d <- data.frame(x1 = x1, x2 = x2)
  d$y <- 2 + 1.8 * x1 - 0.4 * x2 + rnorm(n, sd = 0.2)
  d
}

.orthogonal_enet_reference <- function(d, lambda, alpha) {
  X <- stats::model.matrix(~ x1 + x2, data = d)
  intercept <- mean(d$y)
  yy <- d$y - intercept
  slope <- vapply(2:3, function(j) {
    curvature <- sum(X[, j]^2) / nrow(X)
    partial <- sum(X[, j] * yy) / nrow(X)
    .soft_threshold_ref(partial, lambda * alpha) /
      (curvature + lambda * (1 - alpha))
  }, numeric(1))
  c(intercept, slope)
}

.kkt_enet_violation <- function(X, y, beta, weights, offset, lambda,
                                alpha, penalty.factor) {
  ww <- sum(weights)
  resid <- y - offset - drop(X %*% beta)
  grad <- drop(crossprod(X, weights * resid)) / ww
  smooth <- grad - lambda * (1 - alpha) * penalty.factor * beta
  threshold <- lambda * alpha * penalty.factor
  kkt <- ifelse(beta > 0, abs(smooth - threshold),
                ifelse(beta < 0, abs(smooth + threshold),
                       pmax(0, abs(smooth) - threshold)))
  max(kkt)
}

testthat::test_that("Lasso agrees with exact orthogonal-design soft thresholding", {
  d <- .make_orthogonal_enet_data()
  lambda <- 0.55
  fit <- mrvcr::lmnet(y ~ x1 + x2, data = d, pet = "Lasso", lambda = lambda,
                      tol = 1e-11)
  ref <- .orthogonal_enet_reference(d, lambda, 1)
  testthat::expect_equal(unname(coef(fit)), unname(ref), tolerance = 1e-7)
  testthat::expect_identical(fit$pet, "Lasso")
  testthat::expect_equal(fit$alpha, 1)
  testthat::expect_identical(fit$method, "coordinate_descent")
  testthat::expect_true(fit$converged)
  testthat::expect_true(fit$iterations >= 1)
  testthat::expect_lt(fit$kkt_violation, 1e-7)
  testthat::expect_equal(unname(fitted(fit)),
                         unname(drop(stats::model.matrix(~ x1 + x2, d) %*% ref)),
                         tolerance = 1e-7)
})

testthat::test_that("Elastic net agrees with exact orthogonal-design solution", {
  d <- .make_orthogonal_enet_data()
  for (a in c(0.0, 0.35, 0.8, 1.0)) {
    fit <- mrvcr::lmnet(y ~ x1 + x2, data = d, pet = "ElasticNet",
                        lambda = 0.32, alpha = a, tol = 1e-11)
    ref <- .orthogonal_enet_reference(d, 0.32, a)
    testthat::expect_equal(unname(coef(fit)), unname(ref), tolerance = 1e-7)
    testthat::expect_equal(fit$alpha, a)
    testthat::expect_true(fit$converged)
  }
})

testthat::test_that("Ridge coordinate descent and closed-form ridge agree", {
  set.seed(842)
  d <- data.frame(x = rnorm(120), z = rnorm(120))
  d$y <- 0.8 + 2 * d$x - 1.2 * d$z + rnorm(120)
  ridge <- mrvcr::lmnet(y ~ x + z, data = d, pet = "Ridge", lambda = 0.21)
  cd <- mrvcr::lmnet(y ~ x + z, data = d, pet = "Ridge", lambda = 0.21,
                     method = "coordinate_descent", tol = 1e-10)
  enet_zero <- mrvcr::lmnet(y ~ x + z, data = d, pet = "ElasticNet",
                            lambda = 0.21, alpha = 0, tol = 1e-10)
  testthat::expect_identical(ridge$method, "cholesky")
  testthat::expect_true(cd$converged)
  testthat::expect_equal(unname(coef(ridge)), unname(coef(cd)), tolerance = 1e-6)
  testthat::expect_equal(unname(coef(ridge)), unname(coef(enet_zero)), tolerance = 1e-6)
  testthat::expect_equal(ridge$objective, cd$objective, tolerance = 1e-7)
})

testthat::test_that("Weights, offsets and per-coefficient factors satisfy KKT", {
  set.seed(843)
  n <- 95L
  d <- data.frame(x = rnorm(n), z = rnorm(n))
  d$w <- sample(c(0, 0.3, 1, 2), n, TRUE)
  d$off <- 0.5 + 0.2*d$x
  d$y <- -1 + 1.2*d$x - 0.3*d$z + d$off + rnorm(n, sd = 0.25)
  pf <- c(0, 0.7, 1.4)
  fit <- mrvcr::lmnet(y ~ x + z, data = d, weights = w, offset = off,
                      pet = "ElasticNet", lambda = 0.12, alpha = 0.75,
                      penalty.factor = pf, tol = 1e-10)
  X <- stats::model.matrix(~ x + z, d)
  testthat::expect_true(fit$converged)
  testthat::expect_lt(.kkt_enet_violation(X, d$y, unname(coef(fit)), d$w,
                                          d$off, 0.12, 0.75, pf), 1e-6)
  loss <- sum(d$w * residuals(fit)^2) / (2 * sum(d$w))
  penalty <- 0.12 * sum(pf * (0.75 * abs(coef(fit)) +
                                  (1-0.75)/2 * coef(fit)^2))
  testthat::expect_equal(fit$objective, loss + penalty, tolerance = 1e-9)
  d$w2 <- 7 * d$w
  scaled <- mrvcr::lmnet(y ~ x + z, data = d, weights = w2, offset = off,
                         pet = "ElasticNet", lambda = 0.12, alpha = 0.75,
                         penalty.factor = pf, tol = 1e-10)
  testthat::expect_equal(unname(coef(fit)), unname(coef(scaled)), tolerance = 1e-6)
  testthat::expect_error(predict(fit, d[1:5, ]), "Provide 'offset'")
  testthat::expect_equal(as.vector(predict(fit, d[1:5, ], offset = d$off[1:5])),
                         as.vector(X[1:5, ] %*% coef(fit) + d$off[1:5]),
                         tolerance = 1e-7)
})

testthat::test_that("Multiple responses are fit independently", {
  set.seed(844)
  d <- data.frame(x = rnorm(110), z = rnorm(110))
  d$a <- 1 + 1.4*d$x + rnorm(110, sd = .3)
  d$b <- -2 - .8*d$z + rnorm(110, sd = .3)
  fit <- mrvcr::lmnet(cbind(a,b) ~ x + z, data=d, pet="ElasticNet",
                      lambda=.18, alpha=.6, tol=1e-9)
  fa <- mrvcr::lmnet(a ~ x + z, data=d, pet="ElasticNet",
                     lambda=.18, alpha=.6, tol=1e-9)
  fb <- mrvcr::lmnet(b ~ x + z, data=d, pet="ElasticNet",
                     lambda=.18, alpha=.6, tol=1e-9)
  testthat::expect_true(fit$converged)
  testthat::expect_equal(unname(coef(fit)[,1]), unname(coef(fa)), tolerance=1e-6)
  testthat::expect_equal(unname(coef(fit)[,2]), unname(coef(fb)), tolerance=1e-6)
  testthat::expect_equal(unname(fitted(fit)[,1]), unname(fitted(fa)), tolerance=1e-6)
  testthat::expect_equal(unname(predict(fit, d[1:6,])),
                         unname(cbind(predict(fa, d[1:6,]), predict(fb, d[1:6,]))),
                         tolerance=1e-6)
})

testthat::test_that("Lasso zero lambda converges to ordinary LS on full rank X", {
  set.seed(845)
  d <- data.frame(x = rnorm(90), z = rnorm(90))
  d$y <- 2 + d$x + d$z + rnorm(90)
  fit <- mrvcr::lmnet(y ~ x + z, data=d, pet="Lasso", lambda=0, tol=1e-9)
  ref <- stats::lm(y ~ x + z, data=d)
  testthat::expect_true(fit$converged)
  testthat::expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance=1e-6)
})

testthat::test_that("PET, method, and numerical controls are validated", {
  d <- data.frame(x = 1:10, y = seq(1, 2, length.out=10))
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, pet="ElasticNet", lambda=.1), "alpha")
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, pet="ElasticNet", lambda=.1,
                                     alpha=1.2), "alpha")
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, pet="Lasso", lambda=.1,
                                     alpha=0), "alpha")
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, pet="Lasso", lambda=.1,
                                     method="cholesky"), "coordinate_descent")
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, pet="ElasticNet", lambda=.1,
                                     alpha=.5, method="qr"), "coordinate_descent")
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, pet="Lasso", lambda=.1,
                                     max_iter=0), "max_iter")
  testthat::expect_error(mrvcr::lmnet(y ~ x, data=d, pet="Lasso", lambda=.1,
                                     tol=0), "tol")
})

testthat::test_that("Nonconvergence is reported rather than silently succeeding", {
  set.seed(846)
  d <- data.frame(x = rnorm(80), z = rnorm(80))
  d$y <- 2 + 3*d$x - 1.2*d$z + rnorm(80, sd=.2)
  testthat::expect_warning(
    fit <- mrvcr::lmnet(y ~ x + z, data=d, pet="Lasso", lambda=.01,
                        max_iter=1, tol=1e-15),
    "did not satisfy"
  )
  testthat::expect_false(fit$converged)
  testthat::expect_equal(fit$iterations, 1)
  testthat::expect_gt(fit$kkt_violation, 0)
})

testthat::test_that("Lasso handles high-dimensional dense X with strong shrinkage", {
  set.seed(847)
  n <- 28L
  p <- 75L
  d <- as.data.frame(matrix(rnorm(n * p), nrow = n))
  names(d) <- paste0("x", seq_len(p))
  d$y <- rnorm(n)
  fit <- mrvcr::lmnet(y ~ ., data=d, pet="Lasso", lambda=2,
                      tol=1e-9)
  testthat::expect_true(fit$converged)
  testthat::expect_equal(length(coef(fit)), p + 1L)
  testthat::expect_true(all(is.finite(coef(fit))))
  testthat::expect_true(is.finite(fit$objective))
  testthat::expect_lt(max(abs(unname(coef(fit)[-1L]))), 1e-8)
})
