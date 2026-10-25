# Internal path builder. This class is deliberately separate from ordinary
# lm/GLM inference and keeps only O(p*q*length(lambda)) path coefficients.
.mrvcr_lmnet_make_path <- function(yy, X, ww, oo, penalty_factor,
                                   pet, lambda, alpha, method, max_iter, tol,
                                   mf, tt, cl, model, x, y, multi, n, p, q) {
  fit <- penalized_lm_path_fit_rust(
    yy, X, ww, oo, unname(penalty_factor), as.character(pet),
    as.numeric(lambda), if (is.null(alpha)) NaN else as.numeric(alpha),
    as.character(method), as.integer(max_iter), as.numeric(tol)
  )
  if (!all(fit$converged)) {
    warning("lmnet path did not converge at every lambda; inspect converged, kkt_violation and iterations.",
            call. = FALSE)
  }
  coef_names <- colnames(X)
  response_names <- if (multi) colnames(yy) %||% paste0("Y", seq_len(q)) else NULL
  lambda_names <- format(lambda, trim = TRUE, scientific = TRUE, digits = 8L)
  if (multi) {
    beta <- array(fit$coefficients, dim = c(p, q, length(lambda)),
                  dimnames = list(coef_names, response_names, lambda_names))
  } else {
    beta <- matrix(fit$coefficients, nrow = p, ncol = length(lambda),
                   dimnames = list(coef_names, lambda_names))
  }
  ans <- list(
    call = cl, terms = tt, coefficients = beta,
    objective = as.numeric(fit$objective),
    weighted.rss = as.numeric(fit$weighted_rss),
    pet = fit$pet, lambda = lambda, alpha = fit$alpha, method = fit$method,
    converged = as.logical(fit$converged),
    iterations = as.integer(fit$iterations),
    kkt_violation = as.numeric(fit$kkt_violation),
    penalty.factor = penalty_factor, weights = ww, offset = oo,
    na.action = attr(mf, "na.action"), contrasts = attr(X, "contrasts"),
    xlevels = stats::.getXlevels(tt, mf),
    has_external_offset = !is.null(as.list(cl)$offset),
    n_responses = q, n_coefficients = p,
    rust = fit
  )
  if (isTRUE(model)) ans$model <- mf
  if (isTRUE(x)) ans$x <- X
  if (isTRUE(y)) ans$y <- yy
  class(ans) <- "mrvcr_lmnet_path"
  ans
}

.mrvcr_path_index <- function(object, s) {
  if (is.null(s)) return(seq_along(object$lambda))
  if (!is.numeric(s) || length(s) == 0L || any(!is.finite(s))) {
    stop("'s' must be a finite numeric lambda value from the fitted path.", call. = FALSE)
  }
  idx <- match(s, object$lambda)
  if (anyNA(idx)) {
    stop("Each 's' must exactly match a fitted lambda; interpolation is not implemented.", call. = FALSE)
  }
  idx
}

#' @export
coef.mrvcr_lmnet_path <- function(object, s = NULL, ...) {
  idx <- .mrvcr_path_index(object, s)
  b <- object$coefficients
  if (object$n_responses == 1L) {
    if (length(idx) == 1L) return(b[, idx, drop = TRUE])
    return(b[, idx, drop = FALSE])
  }
  if (length(idx) == 1L) {
    return(matrix(b[, , idx], nrow = object$n_coefficients,
                  ncol = object$n_responses,
                  dimnames = dimnames(b)[1:2]))
  }
  b[, , idx, drop = FALSE]
}

#' @export
predict.mrvcr_lmnet_path <- function(object, newdata = NULL, s = NULL,
                                     offset = NULL, ...) {
  idx <- .mrvcr_path_index(object, s)
  if (is.null(newdata)) {
    # `$x` partially matches `$xlevels` when the optional design matrix was
    # not retained (x = FALSE). Fetch optional fields with exact names.
    stored_x <- object[["x", exact = TRUE]]
    stored_model <- object[["model", exact = TRUE]]
    if (!is.null(stored_x)) {
      xm <- stored_x
      row_names <- rownames(xm)
    } else if (!is.null(stored_model)) {
      xm <- stats::model.matrix(object$terms, stored_model,
                                contrasts.arg = object$contrasts)
      row_names <- rownames(stored_model)
    } else {
      stop("Refit with model = TRUE or x = TRUE to predict without newdata.",
           call. = FALSE)
    }
    off <- if (is.null(offset)) object$offset else offset
    if (is.null(off)) off <- rep.int(0, nrow(xm))
  } else {
    tt <- stats::delete.response(object$terms)
    mf <- stats::model.frame(tt, newdata, na.action = stats::na.pass,
                             xlev = object$xlevels)
    xm <- stats::model.matrix(tt, mf, contrasts.arg = object$contrasts)
    row_names <- rownames(mf)
    off <- if (is.null(offset)) stats::model.offset(mf) else offset
    if (is.null(off) && isTRUE(object$has_external_offset)) {
      stop("Provide 'offset' when predicting from a fit with an external offset.",
           call. = FALSE)
    }
    if (is.null(off)) off <- rep.int(0, nrow(xm))
  }
  if (!identical(colnames(xm), dimnames(object$coefficients)[[1L]])) {
    stop("Prediction model matrix does not match fitted coefficient names.",
         call. = FALSE)
  }
  if (!is.numeric(off) || length(off) != nrow(xm) || any(!is.finite(off))) {
    stop("offset must be a finite numeric vector matching prediction rows.",
         call. = FALSE)
  }
  n <- nrow(xm)
  q <- object$n_responses
  out <- array(NA_real_, dim = c(n, q, length(idx)),
               dimnames = list(row_names,
                               if (q == 1L) NULL else dimnames(object$coefficients)[[2L]],
                               format(object$lambda[idx], trim = TRUE, scientific = TRUE, digits = 8L)))
  for (k in seq_along(idx)) {
    b <- if (q == 1L) {
      matrix(object$coefficients[, idx[k]], ncol = 1L)
    } else {
      matrix(object$coefficients[, , idx[k]], nrow = object$n_coefficients, ncol = q)
    }
    pred <- xm %*% b
    for (r in seq_len(q)) out[, r, k] <- pred[, r] + off
  }
  if (length(idx) == 1L && q == 1L) return(stats::setNames(out[, 1L, 1L], row_names))
  if (length(idx) == 1L) {
    return(matrix(out[, , 1L], nrow = n, ncol = q,
                  dimnames = dimnames(out)[1:2]))
  }
  if (q == 1L) {
    z <- out[, 1L, , drop = FALSE]
    dim(z) <- c(n, length(idx))
    dimnames(z) <- list(row_names, dimnames(out)[[3L]])
    return(z)
  }
  out
}

#' @export
fitted.mrvcr_lmnet_path <- function(object, s = NULL, ...) {
  pred <- stats::predict(object, s = s)
  if (!is.null(object$na.action) && length(dim(pred)) <= 2L) {
    pred <- stats::napredict(object$na.action, pred)
  }
  pred
}

#' @export
residuals.mrvcr_lmnet_path <- function(object, s = NULL, ...) {
  yy <- if (!is.null(object$y)) object$y else if (!is.null(object$model)) {
    stats::model.response(object$model, "numeric")
  } else {
    stop("Refit with model = TRUE or y = TRUE to calculate path residuals.",
         call. = FALSE)
  }
  pred <- stats::predict(object, s = s)
  if (is.null(dim(pred))) return(as.vector(yy) - pred)
  if (length(dim(pred)) == 2L && object$n_responses == 1L) {
    return(sweep(-pred, 1L, as.vector(yy), `+`))
  }
  if (length(dim(pred)) == 2L) return(yy - pred)
  out <- pred
  for (k in seq_len(dim(pred)[3L])) out[, , k] <- yy - pred[, , k]
  out
}

#' @export
print.mrvcr_lmnet_path <- function(x, ...) {
  cat("\nGaussian", x$pet, "regularization path\n")
  cat("Lambdas:", length(x$lambda), "| method:", x$method,
      "| converged:", sum(x$converged), "/", length(x$converged), "\n")
  cat("Lambda range:", format(min(x$lambda)), "to", format(max(x$lambda)), "\n")
  invisible(x)
}
