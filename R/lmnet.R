#' Penalized Gaussian Linear Regression (Experimental)
#'
#' Fit Gaussian models using an explicitly chosen parameter-estimation target
#' (`pet`) and a separate numerical method. The current native target is
#' `pet = "Ridge"`. Lasso and elastic net are reserved in the Rust PET enum,
#' but their solvers are not yet implemented.
#'
#' Ridge minimizes, over all response columns, the sum of weighted residual
#' squares divided by `2 * sum(weights)` plus
#' `lambda / 2 * sum(penalty.factor * coefficients^2)`.
#' An intercept is unpenalized by default. Predictors are not standardized.
#'
#' `lmnet()` is intentionally separate from [lm()] until penalized-model
#' compatibility and inference semantics have been established. Its return
#' value does *not* inherit from `lm`: ordinary OLS standard errors and AIC
#' are not applicable without additional definitions.
#'
#' @param formula Model formula, as in [stats::lm()]. May have a multivariate
#'   response using `cbind()`.
#' @param data Optional data frame or environment containing the variables.
#' @param subset Optional row subset.
#' @param weights Nonnegative prior weights; the penalty is normalized by
#'   `sum(weights)` rather than the number of observations.
#' @param na.action Missing-value handling function.
#' @param offset Optional additive offset; common to all response columns.
#' @param pet Statistical estimation target: currently `"Ridge"`.
#' @param lambda Nonnegative statistical regularization parameter.
#' @param alpha Elastic-net L1 fraction (must be zero for Ridge); reserved for
#'   future lasso/elastic-net implementations.
#' @param method Numerical solver: `"cholesky"`, `"qr"` (pivoted), or
#'   `"householder_qr"`. Cholesky is faster but QR is often preferable for
#'   ill-conditioned designs.
#' @param penalty.factor Nonnegative coefficient-specific quadratic penalty
#'   weights in the order of the model-matrix columns; by default all ones
#'   except the intercept, whose factor is zero.
#' @param model,x,y Whether to retain the model frame, design matrix, or
#'   response in the returned object.
#' @param contrasts Optional list of contrast coding settings.
#'
#' @return An object of class `mrvcr_lmnet` with coefficients, fitted values,
#'   residuals, weighted RSS, the optimized objective, and the Rust result.
#' @export
lmnet <- function(formula, data, subset, weights, na.action, offset,
                  pet = "Ridge", lambda, alpha = 0,
                  method = "cholesky", penalty.factor = NULL,
                  model = TRUE, x = FALSE, y = FALSE, contrasts = NULL) {
  cl <- match.call()
  if (missing(lambda)) {
    stop("'lambda' is required for a penalized PET.", call. = FALSE)
  }
  if (length(lambda) != 1L || !is.numeric(lambda)) {
    stop("'lambda' must be a numeric scalar.", call. = FALSE)
  }
  if (length(alpha) != 1L || !is.numeric(alpha)) {
    stop("'alpha' must be a numeric scalar.", call. = FALSE)
  }
  if (length(pet) != 1L || !is.character(pet) || is.na(pet)) {
    stop("'pet' must be a single character string.", call. = FALSE)
  }
  if (length(method) != 1L || !is.character(method) || is.na(method)) {
    stop("'method' must be a single character string.", call. = FALSE)
  }

  mf_call <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "subset", "weights", "na.action", "offset"),
             names(mf_call), 0L)
  mf_call <- mf_call[c(1L, m)]
  mf_call$drop.unused.levels <- TRUE
  mf_call[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf_call, parent.frame())

  tt <- attr(mf, "terms")
  yy <- stats::model.response(mf, "numeric")
  if (!is.numeric(yy)) {
    stop("Gaussian lmnet requires a numeric response.", call. = FALSE)
  }
  multi <- is.matrix(yy)
  n <- NROW(yy)
  q <- if (multi) NCOL(yy) else 1L
  X <- stats::model.matrix(tt, mf, contrasts)
  p <- NCOL(X)
  if (p == 0L || n == 0L || q == 0L) {
    stop("lmnet requires a nonempty design and response.", call. = FALSE)
  }
  ww <- as.vector(stats::model.weights(mf))
  oo <- as.vector(stats::model.offset(mf))

  if (is.null(penalty.factor)) {
    penalty.factor <- rep.int(1.0, p)
    penalty.factor[colnames(X) == "(Intercept)"] <- 0.0
  } else {
    if (!is.numeric(penalty.factor) || length(penalty.factor) != p) {
      stop("'penalty.factor' must have one numeric entry per model-matrix column.",
           call. = FALSE)
    }
  }
  penalty.factor <- as.numeric(penalty.factor)
  names(penalty.factor) <- colnames(X)

  fit <- penalized_lm_fit_rust(
    yy, X, ww, oo, unname(penalty.factor), as.character(pet),
    as.numeric(lambda), as.numeric(alpha), as.character(method)
  )
  coef_names <- colnames(X)
  response_names <- if (multi) {
    colnames(yy) %||% paste0("Y", seq_len(q))
  } else {
    NULL
  }
  if (multi) {
    beta <- matrix(fit$coefficients, nrow = p, ncol = q,
                   dimnames = list(coef_names, response_names))
    fitted <- matrix(fit$fitted_values, nrow = n, ncol = q,
                     dimnames = list(rownames(mf), response_names))
    residuals <- matrix(fit$residuals, nrow = n, ncol = q,
                        dimnames = list(rownames(mf), response_names))
  } else {
    beta <- stats::setNames(fit$coefficients, coef_names)
    fitted <- stats::setNames(fit$fitted_values, rownames(mf))
    residuals <- stats::setNames(fit$residuals, rownames(mf))
  }

  ans <- list(
    call = cl, terms = tt, coefficients = beta,
    fitted.values = fitted, residuals = residuals,
    objective = fit$objective, weighted.rss = fit$weighted_rss,
    pet = fit$pet, lambda = lambda, alpha = alpha, method = fit$method,
    penalty.factor = penalty.factor, weights = ww, offset = oo,
    na.action = attr(mf, "na.action"),
    contrasts = attr(X, "contrasts"),
    xlevels = stats::.getXlevels(tt, mf),
    has_external_offset = !missing(offset),
    rust = fit
  )
  if (isTRUE(model)) ans$model <- mf
  if (isTRUE(x)) ans$x <- X
  if (isTRUE(y)) ans$y <- yy
  class(ans) <- "mrvcr_lmnet"
  ans
}
