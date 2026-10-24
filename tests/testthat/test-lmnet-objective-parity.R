# Independent tests for the defined Gaussian PET objective.
# Avoids coupling correctness to glmnet's response standardization.

.pet_objective_kkt <- function(X, y, b, lambda, alpha) {
  n <- nrow(X)
  resid <- y - drop(X %*% b)
  grad <- drop(crossprod(X, resid)) / n
  smooth <- grad - lambda * (1 - alpha) * c(0, rep(1, ncol(X) - 1L)) * b
  thr <- lambda * alpha * c(0, rep(1, ncol(X) - 1L))
  v <- ifelse(b > 0, abs(smooth - thr),
              ifelse(b < 0, abs(smooth + thr), pmax(0, abs(smooth) - thr)))
  max(v)
}

testthat::test_that("all Gaussian PETs satisfy original-scale KKT with correlated X", {
  set.seed(851)
  n <- 160L
  z <- rnorm(n)
  d <- data.frame(x = z + 0.3 * rnorm(n),
                  u = -0.4 * z + rnorm(n),
                  v = z + 0.6 * rnorm(n))
  d$y <- 1.3 + 2 * d$x - 0.8 * d$u + rnorm(n, sd = 0.7)
  X <- stats::model.matrix(~ x + u + v, d)
  lambda <- 0.18
  for (a in c(0, 0.3, 1)) {
    pet <- if (a == 0) "Ridge" else if (a == 1) "Lasso" else "ElasticNet"
    fit <- mrvcr::lmnet(y ~ x + u + v, d, pet = pet,
                        lambda = lambda, alpha = a, tol = 1e-11)
    testthat::expect_true(fit$converged,
                          info = paste(pet, "failed to converge"))
    kkt <- .pet_objective_kkt(X, d$y, unname(coef(fit)), lambda, a)
    testthat::expect_true(is.finite(kkt) && kkt < 1e-7,
                          info = paste(pet, "violates the declared objective; KKT =", kkt))
  }
})

testthat::test_that("ridge agrees with direct first-order normal equations", {
  set.seed(852)
  n <- 155L
  d <- data.frame(x = rnorm(n), z = rnorm(n), w = rnorm(n))
  d$y <- -0.2 + 1.6 * d$x - 0.9 * d$z + rnorm(n, sd = 1.3)
  X <- stats::model.matrix(~ x + z + w, d)
  lambda <- 0.13
  P <- diag(c(0, 1, 1, 1))
  b_ref <- as.vector(solve(crossprod(X) / n + lambda * P,
                           crossprod(X, d$y) / n))
  for (method in c("cholesky", "qr", "coordinate_descent")) {
    fit <- mrvcr::lmnet(y ~ x + z + w, d, pet = "Ridge",
                        lambda = lambda, method = method, tol = 1e-11)
    testthat::expect_equal(unname(as.vector(coef(fit))), b_ref,
                           tolerance = 1e-7)
  }
})
