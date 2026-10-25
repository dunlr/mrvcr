# Sparse path constructor called only after lmnet_sparse() input validation.
# Store O(ncoef * nresponse * nlambda) coefficient values and per-lambda
# diagnostics. The sparse design is retained only if keep_x = TRUE.
.mrvcr_lmnet_sparse_make_path <- function(x, yy, ww, oo, penalty_factor,
                                          pet, lambda, alpha, method,
                                          intercept, max_iter, tol, keep_x, cl) {
  fit <- penalized_lm_sparse_path_fit_rust(
    yy, as.integer(nrow(x)), x@p, x@i, x@x, ww, oo,
    penalty_factor, as.character(pet), as.numeric(lambda),
    if (is.null(alpha)) NaN else as.numeric(alpha), as.character(method),
    intercept, as.integer(max_iter), as.numeric(tol)
  )
  if (!all(fit$converged)) {
    warning("Sparse path did not converge at every lambda; inspect KKT diagnostics.",
            call. = FALSE)
  }
  n <- nrow(x)
  p <- ncol(x)
  q <- NCOL(yy)
  xnames <- colnames(x)
  if (is.null(xnames)) xnames <- paste0("x", seq_len(p))
  cn <- if (intercept) c("(Intercept)", xnames) else xnames
  responses <- if (is.matrix(yy)) colnames(yy) else NULL
  if (is.null(responses) && q > 1L) responses <- paste0("Y", seq_len(q))
  ln <- format(lambda, trim = TRUE, scientific = TRUE, digits = 8L)
  if (q == 1L) {
    b <- matrix(fit$coefficients, nrow = length(cn), ncol = length(lambda),
                dimnames = list(cn, ln))
  } else {
    b <- array(fit$coefficients, dim = c(length(cn), q, length(lambda)),
               dimnames = list(cn, responses, ln))
  }
  ans <- list(
    call = cl, coefficients = b,
    objective = as.numeric(fit$objective),
    weighted.rss = as.numeric(fit$weighted_rss),
    penalty.value = as.numeric(fit$penalty_value),
    pet = fit$pet, lambda = as.numeric(lambda), alpha = fit$alpha,
    method = fit$method, converged = as.logical(fit$converged),
    iterations = as.integer(fit$iterations),
    kkt_violation = as.numeric(fit$kkt_violation),
    weights = ww, offset = oo, intercept = intercept,
    penalty.factor = stats::setNames(penalty_factor, xnames),
    xnames = xnames, n_responses = q, n_predictors = p,
    n_coefficients = length(cn), n_obs = n,
    keep_x = keep_x
  )
  if (isTRUE(keep_x)) {
    ans$x <- x
    ans$y <- yy
  }
  class(ans) <- "mrvcr_lmnet_sparse_path"
  ans
}

.mrvcr_sparse_path_index <- function(object, s = NULL) {
  if (is.null(s)) return(seq_along(object$lambda))
  if (!is.numeric(s) || length(s) == 0L || any(!is.finite(s))) {
    stop("'s' must consist of finite numeric fitted lambda values.", call. = FALSE)
  }
  idx <- match(s, object$lambda)
  if (anyNA(idx)) {
    stop("Each 's' must exactly match a fitted lambda; interpolation is unsupported.",
         call. = FALSE)
  }
  idx
}

#' Extract Coefficients from a Sparse Gaussian Regularization Path
#'
#' @param object An `mrvcr_lmnet_sparse_path` object.
#' @param s One or more exactly fitted lambda values; `NULL` selects the whole path.
#' @param ... Unused.
#' @return Coefficient vector, matrix, or array; response and lambda dimensions
#'   are retained where appropriate.
#' @export
coef.mrvcr_lmnet_sparse_path <- function(object, s = NULL, ...) {
  idx <- .mrvcr_sparse_path_index(object, s)
  b <- object$coefficients
  if (object$n_responses == 1L) {
    if (length(idx) == 1L) return(b[, idx, drop = TRUE])
    return(b[, idx, drop = FALSE])
  }
  if (length(idx) == 1L) {
    return(matrix(b[, , idx], nrow = object$n_coefficients,
                  ncol = object$n_responses, dimnames = dimnames(b)[1:2]))
  }
  b[, , idx, drop = FALSE]
}

#' Predict from a Sparse Gaussian Regularization Path
#'
#' Matrix multiplication is carried out with `newx` in its supplied form;
#' a `dgCMatrix` is never explicitly converted to dense. Predictions at a
#' new design require a matching `offset` if the fit used an external offset.
#'
#' @param object An `mrvcr_lmnet_sparse_path`.
#' @param newx A `dgCMatrix` or dense numeric matrix of predictors. If `NULL`,
#'   requires the fit to have been constructed with `keep_x = TRUE`.
#' @param s Fitted lambda values, or `NULL` for all.
#' @param offset Optional vector of row offsets. If `newx` is supplied and the
#'   model was fitted with offsets, the offset must be supplied explicitly.
#' @param ... Unused.
#' @return A vector for one response/one lambda, a matrix for one free
#'   dimension, or an n-by-response-by-lambda array.
#' @export
predict.mrvcr_lmnet_sparse_path <- function(object, newx = NULL, s = NULL,
                                             offset = NULL, ...) {
  idx <- .mrvcr_sparse_path_index(object, s)
  if (is.null(newx)) {
    xm <- object[["x", exact = TRUE]]
    if (is.null(xm)) {
      stop("Supply 'newx' or refit with keep_x = TRUE for training predictions.",
           call. = FALSE)
    }
    off <- if (is.null(offset)) object$offset else offset
  } else {
    xm <- newx
    off <- offset
    if (is.null(off) && !is.null(object$offset)) {
      stop("Provide 'offset' with newx because the fit used an external offset.",
           call. = FALSE)
    }
  }
  if (!inherits(xm, "dgCMatrix") && !is.matrix(xm)) {
    stop("'newx' must be a dgCMatrix or dense numeric matrix.", call. = FALSE)
  }
  if (ncol(xm) != object$n_predictors ||
      (!is.null(colnames(xm)) && !identical(colnames(xm), object$xnames))) {
    stop("'newx' has the wrong number or names of predictor columns.", call. = FALSE)
  }
  if (is.matrix(xm) && (!is.numeric(xm) || any(!is.finite(xm)))) {
    stop("Dense 'newx' must be finite and numeric.", call. = FALSE)
  }
  if (inherits(xm, "dgCMatrix") && any(!is.finite(xm@x))) {
    stop("Sparse 'newx' must have finite nonzero values.", call. = FALSE)
  }
  n <- nrow(xm)
  if (is.null(off)) off <- rep.int(0.0, n)
  if (!is.numeric(off) || length(off) != n || any(!is.finite(off))) {
    stop("'offset' must be finite and match nrow(newx).", call. = FALSE)
  }
  q <- object$n_responses
  ncoef <- object$n_coefficients
  rn <- rownames(xm)
  response_names <- if (q == 1L) NULL else dimnames(object$coefficients)[[2L]]
  lambda_names <- format(object$lambda[idx], trim = TRUE,
                         scientific = TRUE, digits = 8L)
  out <- array(0.0, dim = c(n, q, length(idx)),
               dimnames = list(rn, response_names, lambda_names))
  for (k in seq_along(idx)) {
    b <- if (q == 1L) {
      matrix(object$coefficients[, idx[k]], nrow = ncoef, ncol = 1L)
    } else {
      matrix(object$coefficients[, , idx[k]], nrow = ncoef, ncol = q)
    }
    intercept <- if (object$intercept) as.numeric(b[1L, ]) else rep.int(0.0, q)
    slopes <- b[seq.int(as.integer(object$intercept) + 1L, ncoef), , drop = FALSE]
    # CSC %*% dense coefficients produces only the dense prediction matrix,
    # never a dense copy of the predictor matrix.
    pred <- as.matrix(xm %*% slopes)
    pred <- sweep(pred, 2L, intercept, `+`)
    pred <- sweep(pred, 1L, off, `+`)
    out[, , k] <- pred
  }
  if (length(idx) == 1L && q == 1L) {
    return(stats::setNames(as.vector(out[, 1L, 1L]), rn))
  }
  if (length(idx) == 1L) {
    return(matrix(out[, , 1L], nrow = n, ncol = q,
                  dimnames = list(rn, response_names)))
  }
  if (q == 1L) {
    z <- matrix(out[, 1L, ], nrow = n, ncol = length(idx),
                dimnames = list(rn, lambda_names))
    return(z)
  }
  out
}

#' @export
fitted.mrvcr_lmnet_sparse_path <- function(object, s = NULL, ...) {
  stats::predict(object, s = s)
}

#' @export
residuals.mrvcr_lmnet_sparse_path <- function(object, s = NULL, ...) {
  yy <- object[["y", exact = TRUE]]
  if (is.null(yy)) {
    stop("Refit with keep_x = TRUE to compute training residuals.", call. = FALSE)
  }
  pred <- stats::predict(object, s = s)
  if (is.null(dim(pred))) return(as.vector(yy) - pred)
  if (object$n_responses == 1L) {
    return(sweep(-pred, 1L, as.vector(yy), `+`))
  }
  if (length(dim(pred)) == 2L) return(yy - pred)
  out <- pred
  for (k in seq_len(dim(out)[3L])) out[, , k] <- yy - pred[, , k]
  out
}

#' @export
print.mrvcr_lmnet_sparse_path <- function(x, ...) {
  cat("\nGaussian sparse CSC", x$pet, "regularization path\n")
  cat("Lambdas:", length(x$lambda), "| method:", x$method,
      "| converged:", sum(x$converged), "/", length(x$converged), "\n")
  cat("Lambda range:", format(min(x$lambda)), "to", format(max(x$lambda)), "\n")
  invisible(x)
}
