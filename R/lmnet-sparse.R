#' Gaussian Penalized Regression on Native Sparse CSC Matrices (Experimental)
#'
#' Fits a *single* Gaussian ridge, lasso, or elastic-net penalty to an existing
#' `Matrix::dgCMatrix`, keeping its column-compressed storage intact. The
#' intercept is represented implicitly in the Rust coordinate-descent solver;
#' it is never appended as a column of ones. Sparse predictors are **not**
#' centered, standardized, or converted to a dense matrix.
#'
#' The objective is `sum(w * (y - offset - intercept - x %*% beta)^2) /
#' (2 * sum(w)) + lambda * sum(penalty.factor * (alpha * abs(beta) +
#' (1-alpha)/2 * beta^2))`, summed across response columns. An intercept is
#' never penalized. For multiple responses the penalty is separable, not a
#' group penalty. See [lmnet()] for PET and penalty conventions.
#'
#' This matrix-oriented entry point deliberately does not modify the
#' formula interface of [lmnet()]. Sparse paths and formula-based sparse model
#' matrix construction will be implemented separately.
#'
#' @param x A canonical `Matrix::dgCMatrix` with numeric double values.
#' @param y Finite numeric response vector or matrix (rows = observations).
#' @param pet One of `"Ridge"`, `"Lasso"`, or `"ElasticNet"`.
#' @param lambda One finite nonnegative scalar; paths not yet implemented here.
#' @param alpha `NULL` for ridge/lasso, required for elastic net.
#' @param weights Optional nonnegative observation weights.
#' @param offset Optional row-wise additive offset (shared across responses).
#' @param penalty.factor Optional nonnegative factors for the **columns of x**;
#'   do not include the intercept.
#' @param intercept Whether to estimate an unpenalized intercept.
#' @param method `"auto"` or `"coordinate_descent"` (aliases `"cd"`).
#' @param max_iter Maximum coordinate-descent sweeps per response.
#' @param tol Positive KKT stopping tolerance.
#' @param keep_x Retain the original sparse design for `predict(fit, newx=...)`
#'   checks and reproducibility. No matrix copy is created by this switch.
#'
#' @return An `mrvcr_lmnet_sparse` fit with coefficients, fitted values,
#'   residuals, objective and KKT diagnostics.
#' @export
lmnet_sparse <- function(x, y, pet = "Ridge", lambda, alpha = NULL,
                         weights = NULL, offset = NULL, penalty.factor = NULL,
                         intercept = TRUE, method = "auto", max_iter = 10000L,
                         tol = 1e-8, keep_x = FALSE) {
  if (!inherits(x, "dgCMatrix")) {
    stop("'x' must be a Matrix::dgCMatrix; implicit conversion is disabled.", call. = FALSE)
  }
  if (length(dim(x)) != 2L || nrow(x) == 0L || ncol(x) == 0L) {
    stop("'x' must have positive dimensions.", call. = FALSE)
  }
  if (!is.numeric(y) || !(is.null(dim(y)) || is.matrix(y)) ||
      NROW(y) != nrow(x) || any(!is.finite(y)) || NCOL(y) < 1L) {
    stop("'y' must be a finite numeric vector or matrix with nrow(x) rows.", call. = FALSE)
  }
  if (!is.numeric(lambda) || length(lambda) != 1L || !is.finite(lambda) || lambda < 0) {
    stop("Sparse fitting currently requires one finite nonnegative lambda.", call. = FALSE)
  }
  if (!is.null(alpha) && (!is.numeric(alpha) || length(alpha) != 1L ||
                          !is.finite(alpha))) {
    stop("'alpha' must be NULL or a finite numeric scalar.", call. = FALSE)
  }
  if (!is.character(pet) || length(pet) != 1L || is.na(pet) ||
      !is.character(method) || length(method) != 1L || is.na(method)) {
    stop("'pet' and 'method' must be single character strings.", call. = FALSE)
  }
  if (!is.logical(intercept) || length(intercept) != 1L || is.na(intercept)) {
    stop("'intercept' must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.logical(keep_x) || length(keep_x) != 1L || is.na(keep_x)) {
    stop("'keep_x' must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.numeric(max_iter) || length(max_iter) != 1L ||
      !is.finite(max_iter) || max_iter < 1 || max_iter != floor(max_iter) ||
      max_iter > .Machine$integer.max) {
    stop("'max_iter' must be a positive integer.", call. = FALSE)
  }
  if (!is.numeric(tol) || length(tol) != 1L || !is.finite(tol) || tol <= 0) {
    stop("'tol' must be finite and positive.", call. = FALSE)
  }
  n <- nrow(x)
  p <- ncol(x)
  if (!is.null(weights) && (!is.numeric(weights) || length(weights) != n ||
                            any(!is.finite(weights)) || any(weights < 0) ||
                            !is.finite(sum(weights)) || sum(weights) <= 0)) {
    stop("'weights' must be finite, nonnegative, and sum to a positive finite number.", call. = FALSE)
  }
  if (!is.null(offset) && (!is.numeric(offset) || length(offset) != n ||
                           any(!is.finite(offset)))) {
    stop("'offset' must contain nrow(x) finite numeric values.", call. = FALSE)
  }
  if (is.null(penalty.factor)) penalty.factor <- rep.int(1.0, p)
  if (!is.numeric(penalty.factor) || length(penalty.factor) != p ||
      any(!is.finite(penalty.factor)) || any(penalty.factor < 0)) {
    stop("'penalty.factor' must have one nonnegative finite number per x column.", call. = FALSE)
  }
  if (!is.double(x@x)) {
    stop("CSC x slot must be double-precision numeric.", call. = FALSE)
  }
  yy <- y
  storage.mode(yy) <- "double"
  ww <- if (is.null(weights)) NULL else as.numeric(weights)
  oo <- if (is.null(offset)) NULL else as.numeric(offset)
  res <- penalized_lm_sparse_fit_rust(
    yy, as.integer(n), x@p, x@i, x@x, ww, oo,
    as.numeric(penalty.factor), as.character(pet), as.numeric(lambda),
    if (is.null(alpha)) NaN else as.numeric(alpha), as.character(method),
    intercept, as.integer(max_iter), as.numeric(tol)
  )
  if (!isTRUE(res$converged)) {
    warning("Sparse coordinate descent did not converge to the requested tolerance; inspect the KKT diagnostics.",
            call. = FALSE)
  }
  q <- NCOL(yy)
  xnames <- colnames(x)
  if (is.null(xnames)) xnames <- paste0("x", seq_len(p))
  cn <- if (intercept) c("(Intercept)", xnames) else xnames
  rn <- rownames(x)
  responses <- if (is.matrix(yy)) colnames(yy) else NULL
  if (is.null(responses) && q > 1L) responses <- paste0("Y", seq_len(q))
  if (q == 1L) {
    b <- stats::setNames(as.numeric(res$coefficients), cn)
    fitted <- stats::setNames(as.numeric(res$fitted_values), rn)
    residuals <- stats::setNames(as.numeric(res$residuals), rn)
  } else {
    b <- matrix(res$coefficients, nrow = length(cn), ncol = q,
                dimnames = list(cn, responses))
    fitted <- matrix(res$fitted_values, nrow = n, ncol = q,
                     dimnames = list(rn, responses))
    residuals <- matrix(res$residuals, nrow = n, ncol = q,
                        dimnames = list(rn, responses))
  }
  ans <- list(
    call = match.call(), coefficients = b,
    fitted.values = fitted, residuals = residuals,
    objective = res$objective, weighted.rss = res$weighted_rss,
    penalty.value = res$penalty_value,
    pet = res$pet, lambda = lambda, alpha = res$alpha,
    method = res$method, converged = res$converged,
    iterations = res$iterations, kkt_violation = res$kkt_violation,
    weights = ww, offset = oo, intercept = intercept,
    penalty.factor = stats::setNames(as.numeric(penalty.factor), xnames),
    xnames = xnames, n_responses = q, n_predictors = p
  )
  if (isTRUE(keep_x)) ans$x <- x
  class(ans) <- "mrvcr_lmnet_sparse"
  ans
}

#' @export
coef.mrvcr_lmnet_sparse <- function(object, ...) object$coefficients

#' @export
fitted.mrvcr_lmnet_sparse <- function(object, ...) object$fitted.values

#' @export
residuals.mrvcr_lmnet_sparse <- function(object, ...) object$residuals

#' @export
predict.mrvcr_lmnet_sparse <- function(object, newx = NULL, offset = NULL, ...) {
  if (is.null(newx)) {
    if (!is.null(offset)) {
      stop("An overriding 'offset' requires 'newx'.", call. = FALSE)
    }
    return(stats::fitted(object))
  }
  if (!inherits(newx, "dgCMatrix") && !is.matrix(newx)) {
    stop("'newx' must be a dgCMatrix or dense numeric matrix.", call. = FALSE)
  }
  if (ncol(newx) != object$n_predictors ||
      (!is.null(colnames(newx)) &&
       !identical(colnames(newx), object$xnames))) {
    stop("'newx' has the wrong number or names of predictor columns.", call. = FALSE)
  }
  if (is.matrix(newx) && (!is.numeric(newx) || any(!is.finite(newx)))) {
    stop("Dense 'newx' must be finite and numeric.", call. = FALSE)
  }
  if (inherits(newx, "dgCMatrix") && any(!is.finite(newx@x))) {
    stop("Sparse 'newx' must have finite nonzero values.", call. = FALSE)
  }
  n <- nrow(newx)
  if (is.null(offset)) {
    if (!is.null(object$offset)) {
      stop("Provide 'offset' for newx when the fit used an external offset.",
           call. = FALSE)
    }
    offset <- rep.int(0.0, n)
  }
  if (!is.numeric(offset) || length(offset) != n || any(!is.finite(offset))) {
    stop("'offset' must be finite and match nrow(newx).", call. = FALSE)
  }
  beta <- object$coefficients
  if (is.null(dim(beta))) beta <- matrix(beta, ncol = 1L)
  intercept <- if (object$intercept) as.numeric(beta[1L, ]) else rep.int(0, ncol(beta))
  slopes <- beta[seq.int(as.integer(object$intercept) + 1L, nrow(beta)), , drop = FALSE]
  pred <- as.matrix(newx %*% slopes)
  pred <- sweep(pred, 2L, intercept, `+`)
  pred <- sweep(pred, 1L, offset, `+`)
  if (object$n_responses == 1L) {
    return(stats::setNames(as.vector(pred), rownames(newx)))
  }
  dimnames(pred) <- list(rownames(newx), colnames(object$coefficients))
  pred
}

#' @export
print.mrvcr_lmnet_sparse <- function(x, ...) {
  cat("\nGaussian sparse CSC", x$pet, "fit\n")
  cat("Lambda:", x$lambda, "| Method:", x$method,
      "| Converged:", x$converged, "| Sweeps:", x$iterations, "\n")
  cat("KKT violation:", format(x$kkt_violation), "\n\n")
  print(x$coefficients)
  invisible(x)
}
