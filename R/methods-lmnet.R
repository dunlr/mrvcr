#' @export
coef.mrvcr_lmnet <- function(object, ...) {
  object$coefficients
}

#' @export
fitted.mrvcr_lmnet <- function(object, ...) {
  out <- object$fitted.values
  if (!is.null(object$na.action)) out <- stats::napredict(object$na.action, out)
  out
}

#' @export
residuals.mrvcr_lmnet <- function(object, ...) {
  out <- object$residuals
  if (!is.null(object$na.action)) out <- stats::naresid(object$na.action, out)
  out
}

#' @export
predict.mrvcr_lmnet <- function(object, newdata = NULL, offset = NULL, ...) {
  if (is.null(newdata)) {
    return(stats::fitted(object))
  }
  tt <- stats::delete.response(object$terms)
  mf <- stats::model.frame(tt, newdata, na.action = stats::na.pass,
                           xlev = object$xlevels)
  xm <- stats::model.matrix(tt, mf, contrasts.arg = object$contrasts)
  beta <- object$coefficients
  if (!identical(colnames(xm), if (is.matrix(beta)) rownames(beta) else names(beta))) {
    stop("newdata model matrix does not match fitted coefficients.", call. = FALSE)
  }
  from_formula <- stats::model.offset(mf)
  if (is.null(offset) && !is.null(from_formula)) offset <- from_formula
  if (is.null(offset) && isTRUE(object$has_external_offset)) {
    stop("Provide 'offset' for predictions from a fit using an external offset.",
         call. = FALSE)
  }
  if (is.null(offset)) offset <- rep.int(0.0, NROW(xm))
  if (!is.numeric(offset) || length(offset) != NROW(xm) ||
      any(!is.finite(offset))) {
    stop("'offset' must be a finite numeric vector matching newdata rows.",
         call. = FALSE)
  }
  pred <- xm %*% beta
  if (is.matrix(beta)) {
    pred <- sweep(pred, 1L, offset, `+`)
    colnames(pred) <- colnames(beta)
    return(pred)
  }
  stats::setNames(as.vector(pred) + offset, rownames(mf))
}

#' @export
print.mrvcr_lmnet <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("\nCall:\n")
  print(x$call)
  cat("\nPET:", x$pet, "(lambda =", format(x$lambda),
      ", alpha =", format(x$alpha), ")\n")
  cat("Method:", x$method, "\n")
  if (identical(x$method, "coordinate_descent")) {
    cat("Converged:", x$converged, "| Sweeps:", x$iterations,
        "| KKT violation:", signif(x$kkt_violation, 4L), "\n")
  }
  cat("\nCoefficients:\n")
  print(x$coefficients, digits = digits)
  invisible(x)
}
