# Formula adapter for the existing native CSC coordinate descent backend.
# Only the p-by-n CSC design is ever constructed; the implicit intercept is
# removed in sparse form before passing x@p/x@i/x@x to the Rust solver.
.mrvcr_sparse_formula_design <- function(tt, mf, contrasts = NULL) {
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("sparse = TRUE requires the suggested package 'Matrix'.", call. = FALSE)
  }
  Xfull <- Matrix::sparse.model.matrix(
    tt, data = mf, contrasts.arg = contrasts,
    drop.unused.levels = FALSE
  )
  intercept <- isTRUE(attr(tt, "intercept") == 1L)
  cn <- colnames(Xfull)
  if (intercept && (length(cn) < 1L || !identical(cn[1L], "(Intercept)"))) {
    stop("The sparse model matrix has an unexpected intercept layout.", call. = FALSE)
  }
  X <- if (intercept) Xfull[, -1L, drop = FALSE] else Xfull
  if (ncol(X) < 1L) {
    stop("Sparse formula regression requires at least one predictor column; intercept-only models are not yet supported.",
         call. = FALSE)
  }
  if (!inherits(X, "dgCMatrix")) {
    # Convert between SPARSE formats only. There is deliberately no as.matrix().
    X <- methods::as(X, "dgCMatrix")
  }
  observed_contrasts <- attr(Xfull, "contrasts")
  list(x = X, full_names = cn, intercept = intercept,
       contrasts = if (is.null(observed_contrasts)) contrasts else observed_contrasts)
}

.mrvcr_lmnet_fit_sparse_formula <- function(mf, tt, yy, ww, oo, pet,
                                             lambda, alpha, method,
                                             penalty_factor, max_iter, tol,
                                             model, x, y, contrasts, cl) {
  d <- .mrvcr_sparse_formula_design(tt, mf, contrasts = contrasts)
  p_full <- length(d$full_names)
  if (is.null(penalty_factor)) {
    pf <- rep.int(1.0, ncol(d$x))
  } else {
    if (!is.numeric(penalty_factor) || length(penalty_factor) != p_full ||
        any(!is.finite(penalty_factor)) || any(penalty_factor < 0)) {
      stop("With sparse = TRUE, 'penalty.factor' must contain one finite nonnegative value per full model-matrix column (including the intercept, if present).",
           call. = FALSE)
    }
    if (d$intercept && penalty_factor[1L] != 0) {
      stop("Sparse regression uses an implicit unpenalized intercept; its penalty.factor must be zero.",
           call. = FALSE)
    }
    pf <- if (d$intercept) penalty_factor[-1L] else penalty_factor
  }
  fit <- lmnet_sparse(
    x = d$x, y = yy, pet = pet, lambda = lambda, alpha = alpha,
    weights = ww, offset = oo, penalty.factor = pf,
    intercept = d$intercept, method = method,
    max_iter = max_iter, tol = tol, keep_x = isTRUE(x)
  )
  fit$call <- cl
  fit$terms <- tt
  fit$na.action <- attr(mf, "na.action")
  fit$contrasts <- d$contrasts
  fit$xlevels <- stats::.getXlevels(tt, mf)
  fit$has_external_offset <- "offset" %in% names(cl)
  if (isTRUE(model)) fit$model <- mf
  if (isTRUE(y) && is.null(fit[["y", exact = TRUE]])) fit$y <- yy
  fit$sparse_formula <- TRUE
  class(fit) <- c("mrvcr_lmnet_sparse_formula", class(fit))
  fit
}

#' Predict from Formula-Based Sparse Penalized Regression
#'
#' Builds sparse model matrices with the original factor levels and contrast
#' coding. The fitted intercept is implicit and never appended to the CSC
#' matrix. Newdata formula offsets are read from the new model frame. If the
#' original fit used a separately supplied `offset =`, it must be supplied
#' again for `newdata`, and is added to any offset embedded in the formula.
#' New data containing missing or non-finite predictors are rejected; this
#' iteration does not implement NA-filled out-of-sample predictions.
#'
#' @param object A sparse formula fit from `lmnet(..., sparse = TRUE)`.
#' @param newdata Optional data frame containing predictors.
#' @param s For paths, an exactly fitted lambda or a vector of such lambdas.
#' @param offset External additive offset for newdata, if needed. Any offset
#'   term specified inside the formula is added automatically.
#' @param ... Unused.
#' @return Predictions as a vector, matrix, or array depending on response and
#'   lambda dimensions.
#' @export
predict.mrvcr_lmnet_sparse_formula <- function(object, newdata = NULL,
                                                s = NULL, offset = NULL, ...) {
  path <- inherits(object, "mrvcr_lmnet_sparse_path")
  if (!path && !is.null(s)) {
    stop("'s' is only meaningful for regularization paths.", call. = FALSE)
  }
  if (is.null(newdata)) {
    if (!is.null(offset)) {
      stop("An overriding 'offset' requires 'newdata'.", call. = FALSE)
    }
    if (!path) return(stats::fitted(object))
    X <- object[["x", exact = TRUE]]
    if (is.null(X)) {
      mf <- object[["model", exact = TRUE]]
      if (is.null(mf)) {
        stop("Predicting the fitted path without newdata requires model = TRUE or x = TRUE.",
             call. = FALSE)
      }
      X <- .mrvcr_sparse_formula_design(
        object$terms, mf, contrasts = object$contrasts
      )$x
    }
    return(predict.mrvcr_lmnet_sparse_path(
      object, newx = X, s = s, offset = object$offset
    ))
  }
  tt <- stats::delete.response(object$terms)
  mf <- stats::model.frame(tt, newdata, na.action = stats::na.pass,
                           xlev = object$xlevels)
  d <- .mrvcr_sparse_formula_design(tt, mf, contrasts = object$contrasts)
  X <- d$x
  if (!identical(colnames(X), object$xnames)) {
    stop("Prediction sparse model matrix does not match fitted predictor columns.",
         call. = FALSE)
  }
  if (any(!is.finite(X@x))) {
    stop("newdata contains missing or non-finite predictor values.",
         call. = FALSE)
  }
  from_formula <- stats::model.offset(mf)
  if (isTRUE(object$has_external_offset) && is.null(offset)) {
    stop("Provide 'offset' for newdata because the model used an external offset.",
         call. = FALSE)
  }
  if (!is.null(offset) && (!is.numeric(offset) || length(offset) != nrow(X) ||
                          any(!is.finite(offset)))) {
    stop("'offset' must be a finite numeric vector matching newdata rows.",
         call. = FALSE)
  }
  off <- if (is.null(offset)) from_formula else if (is.null(from_formula)) {
    offset
  } else {
    offset + from_formula
  }
  if (path) {
    predict.mrvcr_lmnet_sparse_path(object, newx = X, s = s, offset = off)
  } else {
    predict.mrvcr_lmnet_sparse(object, newx = X, offset = off)
  }
}

# Expand na.exclude training predictions along the observation axis; supports
# vector, matrix and 3-D path arrays without flattening the response axes.
.mrvcr_sparse_formula_na_restore <- function(out, action, residual = FALSE) {
  if (is.null(action)) return(out)
  restore <- if (residual) stats::naresid else stats::napredict
  if (length(dim(out)) <= 2L) return(restore(action, out))
  if (!inherits(action, "exclude")) return(out)
  sz <- dim(out)
  n_tot <- sz[1L] + length(action)
  kept <- setdiff(seq_len(n_tot), as.integer(action))
  result <- array(NA_real_, dim = c(n_tot, sz[-1L]),
                  dimnames = c(list(NULL), dimnames(out)[-1L]))
  result[kept, , ] <- out
  result
}

#' @export
fitted.mrvcr_lmnet_sparse_formula <- function(object, s = NULL, ...) {
  out <- if (inherits(object, "mrvcr_lmnet_sparse_path")) {
    predict.mrvcr_lmnet_sparse_formula(object, s = s)
  } else {
    if (!is.null(s)) stop("'s' is only meaningful for paths.", call. = FALSE)
    object$fitted.values
  }
  .mrvcr_sparse_formula_na_restore(out, object$na.action)
}

#' @export
residuals.mrvcr_lmnet_sparse_formula <- function(object, s = NULL, ...) {
  if (inherits(object, "mrvcr_lmnet_sparse_path")) {
    yy <- object[["y", exact = TRUE]]
    if (is.null(yy)) {
      mf <- object[["model", exact = TRUE]]
      if (is.null(mf)) {
        stop("Path residuals require model = TRUE or y = TRUE, and model = TRUE or x = TRUE for predictions.",
             call. = FALSE)
      }
      yy <- stats::model.response(mf, "numeric")
    }
    pred <- predict.mrvcr_lmnet_sparse_formula(object, s = s)
    if (is.null(dim(pred))) {
      out <- as.vector(yy) - pred
    } else if (object$n_responses == 1L) {
      out <- sweep(-pred, 1L, as.vector(yy), `+`)
    } else if (length(dim(pred)) == 2L) {
      out <- yy - pred
    } else {
      out <- pred
      for (k in seq_len(dim(pred)[3L])) out[, , k] <- yy - pred[, , k]
    }
  } else {
    if (!is.null(s)) stop("'s' is only meaningful for paths.", call. = FALSE)
    out <- object$residuals
  }
  .mrvcr_sparse_formula_na_restore(out, object$na.action, residual = TRUE)
}

#' @export
print.mrvcr_lmnet_sparse_formula <- function(x, ...) {
  if (inherits(x, "mrvcr_lmnet_sparse_path")) {
    print.mrvcr_lmnet_sparse_path(x, ...)
  } else {
    print.mrvcr_lmnet_sparse(x, ...)
  }
  invisible(x)
}
