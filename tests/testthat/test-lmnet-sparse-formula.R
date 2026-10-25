# Integration of Matrix::sparse.model.matrix with the existing native CSC fits.
testthat::skip_if_not_installed("Matrix")

.sparse_formula_data <- function(n = 140L, seed = 9473L) {
  set.seed(seed)
  d <- data.frame(
    z = stats::rnorm(n),
    u = stats::rnorm(n),
    group = factor(sample(c("A", "B", "C"), n, replace = TRUE),
                   levels = c("A", "B", "C")),
    w = stats::runif(n, .4, 2.2),
    off = stats::runif(n, -.15, .15),
    ext = stats::runif(n, -.1, .1)
  )
  d$z[sample.int(n, n %/% 2L)] <- 0
  d$u[sample.int(n, 3L * n %/% 4L)] <- 0
  d$y <- 1.3 + .7 * d$z - .4 * d$u +
    ifelse(d$group == "B", .25, ifelse(d$group == "C", -.3, 0)) +
    d$off + d$ext + stats::rnorm(n, sd = .12)
  d$y2 <- -.8 + 1.25 * d$y + stats::rnorm(n, sd = .08)
  d
}

.sparse_formula_coef <- function(object, s = NULL) {
  unname(stats::coef(object, s = s))
}

testthat::test_that("sparse formula builds CSC design and matches dense scalar PETs", {
  d <- .sparse_formula_data()
  f <- y ~ z + u + group
  for (pet in c("Ridge", "Lasso", "ElasticNet")) {
    a <- if (pet == "ElasticNet") .35 else NULL
    dense <- mrvcr::lmnet(f, d, pet = pet, lambda = .04,
                           alpha = a, weights = w, tol = 1e-10)
    sparse <- mrvcr::lmnet(f, d, pet = pet, lambda = .04,
                            alpha = a, weights = w, sparse = TRUE,
                            x = TRUE, tol = 1e-10)
    testthat::expect_s3_class(sparse, "mrvcr_lmnet_sparse_formula")
    testthat::expect_s3_class(sparse, "mrvcr_lmnet_sparse")
    testthat::expect_true(inherits(sparse[["x", exact = TRUE]], "dgCMatrix"))
    testthat::expect_identical(names(stats::coef(sparse)), names(stats::coef(dense)))
    testthat::expect_equal(unname(stats::coef(sparse)), unname(stats::coef(dense)), tolerance = 1e-6)
    testthat::expect_equal(unname(stats::fitted(sparse)), unname(stats::fitted(dense)), tolerance = 1e-6)
    testthat::expect_equal(unname(stats::residuals(sparse)), unname(stats::residuals(dense)), tolerance = 1e-6)
    testthat::expect_equal(unname(stats::predict(sparse, newdata = d[1:7, ])),
                           unname(stats::predict(dense, newdata = d[1:7, ])),
                           tolerance = 1e-6)
    testthat::expect_true(isTRUE(sparse$converged))
  }
})

testthat::test_that("sparse formula paths agree with dense and scalar fits", {
  d <- .sparse_formula_data()
  f <- y ~ z + group + u
  lambda <- c(.35, .2, .09, .03)
  dense <- mrvcr::lmnet(f, data = d, pet = "ElasticNet", alpha = .4,
                         lambda = lambda, tol = 1e-10)
  sparse <- mrvcr::lmnet(f, data = d, pet = "ElasticNet", alpha = .4,
                          lambda = lambda, sparse = TRUE, tol = 1e-10)
  testthat::expect_s3_class(sparse, "mrvcr_lmnet_sparse_path")
  testthat::expect_s3_class(sparse, "mrvcr_lmnet_sparse_formula")
  testthat::expect_null(sparse[["x", exact = TRUE]])
  testthat::expect_true(all(sparse$converged))
  testthat::expect_equal(unname(stats::coef(sparse)), unname(stats::coef(dense)), tolerance = 1e-6)
  testthat::expect_equal(unname(stats::predict(sparse, s = lambda[2L])),
                         unname(stats::predict(dense, s = lambda[2L])), tolerance = 1e-6)
  testthat::expect_equal(unname(stats::predict(sparse, newdata = d[1:6, ], s = lambda[3L])),
                         unname(stats::predict(dense, newdata = d[1:6, ], s = lambda[3L])),
                         tolerance = 1e-6)
  testthat::expect_equal(unname(stats::fitted(sparse, s = lambda[1L])),
                         unname(stats::predict(sparse, s = lambda[1L])), tolerance = 1e-10)
  testthat::expect_equal(unname(stats::residuals(sparse, s = lambda[1L])),
                         unname(d$y - stats::predict(sparse, s = lambda[1L])), tolerance = 1e-8)
  scalar <- mrvcr::lmnet(f, data = d, pet = "ElasticNet", alpha = .4,
                          lambda = lambda[3L], sparse = TRUE, tol = 1e-10)
  testthat::expect_equal(unname(stats::coef(sparse, s = lambda[3L])),
                         unname(stats::coef(scalar)), tolerance = 1e-6)
})

testthat::test_that("formula offset and external offset are both respected", {
  d <- .sparse_formula_data()
  f <- y ~ z + group + offset(off)
  fit <- mrvcr::lmnet(f, data = d, pet = "Ridge", lambda = .08,
                       offset = ext, sparse = TRUE, tol = 1e-11)
  testthat::expect_error(stats::predict(fit, newdata = d[1:4, ]), "offset")
  pred <- stats::predict(fit, newdata = d[1:4, ], offset = d$ext[1:4])
  xx <- stats::model.matrix(y ~ z + group, data = d[1:4, ])
  manual <- as.vector(xx %*% stats::coef(fit)) + d$off[1:4] + d$ext[1:4]
  testthat::expect_equal(unname(pred), manual, tolerance = 1e-7)
  testthat::expect_equal(unname(stats::predict(fit)), unname(stats::fitted(fit)))
  # Separately check that formula-only offsets require no explicit argument.
  without_ext <- mrvcr::lmnet(f, data = d, pet = "Ridge", lambda = .08,
                               sparse = TRUE)
  testthat::expect_length(stats::predict(without_ext, newdata = d[1:4, ]), 4L)
})

testthat::test_that("nonuniform penalty factors map from full model matrix to slopes", {
  d <- .sparse_formula_data()
  f <- y ~ z + group
  dense_x <- stats::model.matrix(f, data = d)
  pf <- c(0, .5, 1.8, 0)
  testthat::expect_identical(length(pf), ncol(dense_x))
  sparse <- mrvcr::lmnet(f, data = d, pet = "ElasticNet", alpha = .3,
                          lambda = .06, penalty.factor = pf, sparse = TRUE)
  dense <- mrvcr::lmnet(f, data = d, pet = "ElasticNet", alpha = .3,
                         lambda = .06, penalty.factor = pf)
  testthat::expect_equal(unname(stats::coef(sparse)), unname(stats::coef(dense)), tolerance = 1e-6)
  testthat::expect_error(mrvcr::lmnet(f, d, pet = "Ridge", lambda = .2,
                                         penalty.factor = c(1, 1, 1, 1), sparse = TRUE),
                         "unpenalized intercept")
  testthat::expect_error(mrvcr::lmnet(f, d, pet = "Ridge", lambda = .2,
                                         penalty.factor = pf[-1L], sparse = TRUE),
                         "full model-matrix")
})

testthat::test_that("explicit factor contrasts are retained for newdata predictions", {
  d <- .sparse_formula_data()
  f <- y ~ group + z
  cc <- list(group = "contr.sum")
  dense <- mrvcr::lmnet(f, data = d, pet = "Ridge", lambda = .035,
                         contrasts = cc)
  sparse <- mrvcr::lmnet(f, data = d, pet = "Ridge", lambda = .035,
                          contrasts = cc, sparse = TRUE)
  testthat::expect_equal(unname(stats::coef(sparse)),
                         unname(stats::coef(dense)), tolerance = 1e-6)
  testthat::expect_equal(unname(stats::predict(sparse, newdata = d[1:7, ])),
                         unname(stats::predict(dense, newdata = d[1:7, ])),
                         tolerance = 1e-6)
})

testthat::test_that("no-intercept factor formulas match dense and keep factor levels", {
  d <- .sparse_formula_data()
  f <- y ~ 0 + group + z
  dense <- mrvcr::lmnet(f, data = d, pet = "Lasso", lambda = .035,
                         tol = 1e-10)
  sparse <- mrvcr::lmnet(f, data = d, pet = "Lasso", lambda = .035,
                          sparse = TRUE, tol = 1e-10)
  testthat::expect_false(sparse$intercept)
  testthat::expect_equal(unname(stats::coef(sparse)), unname(stats::coef(dense)), tolerance = 1e-6)
  testthat::expect_equal(unname(stats::predict(sparse, newdata = d[1:6, ])),
                         unname(stats::predict(dense, newdata = d[1:6, ])), tolerance = 1e-6)
  nd <- d[1:3, ]
  nd$group <- factor(c("A", "B", "C"), levels = levels(d$group))
  testthat::expect_length(stats::predict(sparse, newdata = nd), 3L)
  nd$group <- factor(c("A", "D", "C"))
  testthat::expect_error(stats::predict(sparse, newdata = nd), "new level|new levels|levels")
})

testthat::test_that("na.exclude preserves omission records and training shape", {
  d <- .sparse_formula_data()
  d$z[c(3L, 12L, 20L)] <- NA_real_
  f <- y ~ z + group
  fit <- mrvcr::lmnet(f, d, pet = "Ridge", lambda = .08,
                       na.action = stats::na.exclude, sparse = TRUE)
  testthat::expect_length(stats::fitted(fit), nrow(d))
  testthat::expect_length(stats::residuals(fit), nrow(d))
  testthat::expect_true(all(is.na(stats::fitted(fit)[c(3L, 12L, 20L)])))
  testthat::expect_true(all(is.na(stats::residuals(fit)[c(3L, 12L, 20L)])))
  path <- mrvcr::lmnet(f, d, pet = "Ridge", lambda = c(.2, .08),
                        na.action = stats::na.exclude, sparse = TRUE)
  testthat::expect_identical(dim(stats::fitted(path)), c(as.integer(nrow(d)), 2L))
  testthat::expect_true(all(is.na(stats::fitted(path)[c(3L, 12L, 20L), ])))
})

testthat::test_that("multivariate sparse formula supports coefficients and predictions", {
  d <- .sparse_formula_data()
  f <- cbind(y, y2) ~ z + group
  lam <- c(.16, .07, .025)
  s <- mrvcr::lmnet(f, d, pet = "ElasticNet", alpha = .4,
                     lambda = lam, sparse = TRUE)
  dense <- mrvcr::lmnet(f, d, pet = "ElasticNet", alpha = .4,
                         lambda = lam)
  testthat::expect_identical(dim(stats::coef(s)), dim(stats::coef(dense)))
  testthat::expect_equal(unname(stats::coef(s)), unname(stats::coef(dense)), tolerance = 1e-6)
  testthat::expect_identical(dim(stats::predict(s, newdata = d[1:8, ])), c(8L, 2L, 3L))
  testthat::expect_identical(dim(stats::residuals(s, s = lam[2L])), c(as.integer(nrow(d)), 2L))
})

testthat::test_that("sparse formula storage and dispatch stay explicit", {
  d <- .sparse_formula_data()
  f <- y ~ z + group
  dense <- mrvcr::lmnet(f, d, pet = "Ridge", lambda = .1)
  testthat::expect_s3_class(dense, "mrvcr_lmnet")
  testthat::expect_false(inherits(dense, "mrvcr_lmnet_sparse_formula"))
  fit <- mrvcr::lmnet(f, d, pet = "Ridge", lambda = .1,
                       sparse = TRUE, x = TRUE, model = FALSE)
  testthat::expect_true(inherits(fit[["x", exact = TRUE]], "dgCMatrix"))
  testthat::expect_null(fit[["model", exact = TRUE]])
  path <- mrvcr::lmnet(f, d, pet = "Lasso", lambda = c(.3, .1),
                        sparse = TRUE, model = FALSE, x = FALSE)
  testthat::expect_error(stats::predict(path), "model = TRUE|x = TRUE")
  testthat::expect_equal(dim(stats::predict(path, newdata = d[1:5, ])), c(5L, 2L))
  testthat::expect_error(mrvcr::lmnet(f, d, lambda = .1, sparse = NA), "sparse")
  testthat::expect_error(mrvcr::lmnet(y ~ 1, d, lambda = .1, sparse = TRUE), "intercept-only")
})
