`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

.mrvcr_glm_null_deviance <- function(y, family, weights, offset, intercept) {
  n <- NROW(y)
  y <- as.numeric(y)

  if (is.null(weights)) {
    weights <- rep.int(1, n)
  }

  if (is.null(offset)) {
    offset <- rep.int(0, n)
  }

  if (intercept) {
    if (identical(family$family, "gaussian") &&
        identical(family$link, "identity")) {
      mu0 <- offset + stats::weighted.mean(y - offset, weights)
      return(sum(family$dev.resids(y, mu0, weights)))
    }

    x0 <- matrix(1, nrow = n, ncol = 1)

    null_fit <- stats::glm.fit(
      x = x0,
      y = y,
      weights = weights,
      offset = offset,
      family = family,
      intercept = TRUE
    )

    return(null_fit$deviance)
  }

  mu0 <- family$linkinv(offset)
  sum(family$dev.resids(y, mu0, weights))
}

.mrvcr_glm_aic <- function(y, family, fitted.values, weights, deviance, rank) {
  n_obs <- NROW(y)

  if (is.null(weights)) {
    weights <- rep.int(1, n_obs)
  }

  n_for_family <- rep.int(1, n_obs)

  tryCatch(
    family$aic(y, n_for_family, fitted.values, weights, deviance) + 2 * rank,
    error = function(e) NA_real_
  )
}
