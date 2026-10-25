#' Penalized Gaussian Linear Regression (Experimental)
#'
#' Fits Gaussian models with a parameter estimation target (`pet`) distinct
#' from the numerical solver (`method`). The PET selects a penalized objective:
#'
#' `weighted_RSS / (2 * sum(weights)) + lambda * sum_jr penalty.factor[j] *
#' (alpha * abs(beta[j,r]) + (1-alpha)/2 * beta[j,r]^2)`.
#'
#' `pet = "Ridge"` fixes `alpha = 0`, `pet = "Lasso"` fixes `alpha = 1`,
#' and `pet = "ElasticNet"` requires an explicit `alpha` in `[0, 1]`.
#' `method = "auto"` selects Cholesky for ridge and coordinate descent for
#' lasso/elastic net. A strictly decreasing lambda vector uses warm-start
#' coordinate descent and returns a `mrvcr_lmnet_path` object. No automatic
#' standardization is applied, and coordinate descent does not build X'X.
#' An intercept is unpenalized by default; multiple response columns are
#' penalized independently, not with a joint group penalty.
#'
#' `lmnet()` is separate from [lm()]; returned objects do not inherit the
#' conventional `lm` class, which would imply unjustified OLS inference.
#'
#' @param formula Model formula, as in [stats::lm()]. May have a multivariate
#'   response using `cbind()`.
#' @param data Optional data frame or environment containing the variables.
#' @param subset Optional row subset.
#' @param weights Nonnegative prior weights; the penalty is normalized by
#'   `sum(weights)` rather than the number of observations.
#' @param na.action Missing-value handling function.
#' @param offset Optional additive offset; common to all response columns.
#' @param pet Statistical target: `"Ridge"`, `"Lasso"`, or `"ElasticNet"`.
#' @param lambda Nonnegative regularization parameter (scalar), or a numeric
#'   vector of at least two strictly decreasing values for a warm-start path.
#' @param alpha Elastic-net L1 fraction; omitted for ridge/lasso and required
#'   for elastic net. PET-specific constraints are validated in Rust.
#' @param method Numerical method: `"auto"`, `"cholesky"`, `"qr"`
#'   (pivoted), `"householder_qr"`, or `"coordinate_descent"`.
#' @param penalty.factor Nonnegative coefficient-specific penalty
#'   factors in the order of the model-matrix columns; by default all ones
#'   except the intercept, whose factor is zero.
#' @param max_iter Maximum coordinate-descent sweeps per response.
#' @param tol Relative first-order (KKT) stopping tolerance for coordinate descent.
#' @param model,x,y Whether to retain the model frame, design matrix, or
#'   response in the returned object.
#' @param contrasts Optional list of contrast coding settings.
#'
#' @return A scalar lambda returns class `mrvcr_lmnet` (unchanged). A vector
#'   returns `mrvcr_lmnet_path`: coefficient paths and lambda-specific
#'   diagnostics, with fitted values computed on demand.
#' @export
lmnet <- function(formula, data, subset, weights, na.action, offset,
                  pet = "Ridge", lambda, alpha = NULL,
                  method = "auto", penalty.factor = NULL,
                  model = TRUE, x = FALSE, y = FALSE, contrasts = NULL,
                  max_iter = 10000L, tol = 1e-8) {
  cl <- match.call()
  if (missing(lambda)) {
    stop("'lambda' is required for a penalized PET.", call. = FALSE)
  }
  if (!is.numeric(lambda) || length(lambda) == 0L ||
      any(!is.finite(lambda)) || any(lambda < 0)) {
    stop("'lambda' must contain finite, nonnegative numeric values.", call. = FALSE)
  }
  if (length(lambda) > 1L && any(diff(lambda) >= 0)) {
    stop("A lambda path must be strictly decreasing.", call. = FALSE)
  }
  if (!is.null(alpha) &&
      (length(alpha) != 1L || !is.numeric(alpha) || !is.finite(alpha))) {
    stop("'alpha' must be NULL or a finite numeric scalar.", call. = FALSE)
  }
  if (length(max_iter) != 1L || !is.numeric(max_iter) ||
      !is.finite(max_iter) || max_iter < 1 || max_iter != floor(max_iter) ||
      max_iter > .Machine$integer.max) {
    stop("'max_iter' must be a positive integer.", call. = FALSE)
  }
  if (length(tol) != 1L || !is.numeric(tol) ||
      !is.finite(tol) || tol <= 0) {
    stop("'tol' must be finite and positive.", call. = FALSE)
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

  if (length(lambda) > 1L) {
    return(.mrvcr_lmnet_make_path(
      yy = yy, X = X, ww = ww, oo = oo, penalty_factor = penalty.factor,
      pet = pet, lambda = lambda, alpha = alpha, method = method,
      max_iter = max_iter, tol = tol, mf = mf, tt = tt, cl = cl,
      model = model, x = x, y = y, multi = multi, n = n, p = p, q = q
    ))
  }

  fit <- penalized_lm_fit_rust(
    yy, X, ww, oo, unname(penalty.factor), as.character(pet),
    as.numeric(lambda), if (is.null(alpha)) NaN else as.numeric(alpha),
    as.character(method), as.integer(max_iter), as.numeric(tol)
  )
  if (!isTRUE(fit$converged)) {
    warning("lmnet coordinate descent did not satisfy the requested KKT tolerance; inspect fit$converged, fit$kkt_violation and fit$iterations.",
            call. = FALSE)
  }
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
    pet = fit$pet, lambda = lambda, alpha = fit$alpha, method = fit$method,
    converged = fit$converged, iterations = fit$iterations,
    kkt_violation = fit$kkt_violation,
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
