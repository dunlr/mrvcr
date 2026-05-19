#' @export
coef.mrvcr_lm <- function(object, ...) {
  object$coefficients
}

#' @export
fitted.mrvcr_lm <- function(object, ...) {
  out <- object$fitted.values

  if (!is.null(object$na.action)) {
    out <- stats::napredict(object$na.action, out)
  }

  out
}

#' @export
residuals.mrvcr_lm <- function(object, ...) {
  out <- object$residuals

  if (!is.null(object$na.action)) {
    out <- stats::naresid(object$na.action, out)
  }

  out
}

#' @export
df.residual.mrvcr_lm <- function(object, ...) {
  object$df.residual
}

#' @export
vcov.mrvcr_lm <- function(object, ...) {
  if (inherits(object, "mrvcr_mlm")) {
    stop("vcov.mrvcr_mlm is not implemented yet.", call. = FALSE)
  }

  if (is.null(object$rust) || is.null(object$rust$vcov)) {
    stop(
      "Variance-covariance matrix was not computed; refit with se = TRUE.",
      call. = FALSE
    )
  }

  p <- length(object$coefficients)
  out <- matrix(object$rust$vcov, nrow = p, ncol = p)

  nm <- names(object$coefficients)
  dimnames(out) <- list(nm, nm)

  out
}

#' @export
sigma.mrvcr_lm <- function(object, ...) {
  if (inherits(object, "mrvcr_mlm")) {
    rss <- colSums(object$residuals^2)
    return(sqrt(rss / object$df.residual))
  }

  sqrt(sum(object$residuals^2) / object$df.residual)
}

#' @export
print.mrvcr_lm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  if (identical(x$compat, "stats") && inherits(x, "lm")) {
    return(NextMethod())
  }

  cat("\nCall:\n")
  print(x$call)

  cat("\nCoefficients:\n")
  print(coef(x), digits = digits)

  cat("\nBackend method:", x$method, "\n")
  cat("Compatibility mode:", x$compat, "\n")

  invisible(x)
}

#' @export
summary.mrvcr_lm <- function(object, ...) {
  if (identical(object$compat, "stats") && inherits(object, "lm")) {
    return(NextMethod())
  }

  if (inherits(object, "mrvcr_mlm")) {
    return(summary_mrvcr_mlm(object, ...))
  }

  coef <- object$coefficients
  residuals <- object$residuals
  df <- object$df.residual

  rss <- sum(residuals^2)
  sigma <- sqrt(rss / df)

  if (!is.null(object$rust) && !is.null(object$rust$vcov)) {
    vc <- vcov(object)
    se <- sqrt(diag(vc))
    tval <- coef / se
    pval <- 2 * stats::pt(abs(tval), df = df, lower.tail = FALSE)

    coef_table <- cbind(
      Estimate = coef,
      `Std. Error` = se,
      `t value` = tval,
      `Pr(>|t|)` = pval
    )
  } else {
    coef_table <- cbind(Estimate = coef)
  }

  y <- if (!is.null(object$model)) {
    stats::model.response(object$model)
  } else if (!is.null(object$y)) {
    object$y
  } else {
    NULL
  }

  r_squared <- NA_real_
  adj_r_squared <- NA_real_
  fstatistic <- NULL

  if (!is.null(y) && length(y) == length(residuals)) {
    rss0 <- sum((y - mean(y))^2)
    r_squared <- 1 - rss / rss0

    p <- length(coef)
    n <- length(y)
    adj_r_squared <- 1 - (1 - r_squared) * ((n - 1) / df)

    if (p > 1L && df > 0L) {
      ms_model <- (rss0 - rss) / (p - 1L)
      ms_resid <- rss / df
      fstatistic <- c(
        value = ms_model / ms_resid,
        numdf = p - 1L,
        dendf = df
      )
    }
  }

  ans <- list(
    call = object$call,
    terms = object$terms,
    residuals = residuals,
    coefficients = coef_table,
    sigma = sigma,
    df = c(length(coef), df, length(coef)),
    r.squared = r_squared,
    adj.r.squared = adj_r_squared,
    fstatistic = fstatistic,
    method = object$method,
    compat = object$compat,
    rust = object$rust
  )

  class(ans) <- "summary.mrvcr_lm"
  ans
}

summary_mrvcr_mlm <- function(object, ...) {
  ans <- list(
    call = object$call,
    terms = object$terms,
    coefficients = object$coefficients,
    residuals = object$residuals,
    fitted.values = object$fitted.values,
    sigma = sigma(object),
    df.residual = object$df.residual,
    method = object$method,
    compat = object$compat,
    rust = object$rust
  )

  class(ans) <- "summary.mrvcr_mlm"
  ans
}

#' @export
print.summary.mrvcr_lm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("\nCall:\n")
  print(x$call)

  cat("\nResiduals:\n")
  print(summary(x$residuals), digits = digits)

  cat("\nCoefficients:\n")
  printCoefmat(x$coefficients, digits = digits, signif.stars = TRUE)

  cat("\nResidual standard error:", format(signif(x$sigma, digits)), "\n")

  if (is.finite(x$r.squared)) {
    cat("Multiple R-squared:", format(signif(x$r.squared, digits)), "\n")
    cat("Adjusted R-squared:", format(signif(x$adj.r.squared, digits)), "\n")
  }

  if (!is.null(x$fstatistic)) {
    cat(
      "F-statistic:",
      format(signif(x$fstatistic[["value"]], digits)),
      "on",
      x$fstatistic[["numdf"]],
      "and",
      x$fstatistic[["dendf"]],
      "DF\n"
    )
  }

  cat("Backend method:", x$method, "\n")
  cat("Compatibility mode:", x$compat, "\n")

  invisible(x)
}

#' @export
print.summary.mrvcr_mlm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("\nCall:\n")
  print(x$call)

  cat("\nCoefficients:\n")
  print(x$coefficients, digits = digits)

  cat("\nResidual standard errors by response:\n")
  print(x$sigma, digits = digits)

  cat("\nBackend method:", x$method, "\n")
  cat("Compatibility mode:", x$compat, "\n")

  invisible(x)
}

#' @export
predict.mrvcr_lm <- function(object, newdata = NULL, ...) {
  if (identical(object$compat, "stats") && inherits(object, "lm")) {
    return(NextMethod())
  }

  if (is.null(newdata)) {
    return(fitted(object))
  }

  tt <- delete.response(stats::terms(object))
  mf <- stats::model.frame(tt, newdata, na.action = stats::na.pass, xlev = object$xlevels)
  X <- stats::model.matrix(tt, mf, contrasts.arg = object$contrasts)

  drop(X %*% object$coefficients)
}
