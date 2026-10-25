#' Multivariate Response Variance Component model
#' @param formula Multivariate-response regression formula, e.g. `cbind(y1, y2) ~ x`.
#' @param data A data frame or environment containing model variables.
#' @param pet Requested estimation criterion; the present Rust engine performs
#'   only ordinary least squares, not variance-component ML/REML estimation.
#'
#' @export
mvcm <- function(formula, data = parent.frame(), pet = "MLE") {
  # 1. Process formula elements
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())

  X <- stats::model.matrix(attr(mf, "terms"), mf)

  # Extract the multi-column matrix response block
  Y_matrix <- stats::model.response(mf)
  if (!is.matrix(Y_matrix)) {
    Y_matrix <- as.matrix(Y_matrix)
  }

  nc <- ncol(X)
  n_resp <- ncol(Y_matrix)

  # Current 3-argument extendr wrapper accepts R matrices directly.
  fit_output <- mvcm_fit_rust(Y_matrix, X, as.character(pet))
  if (identical(fit_output$engine, "faer-multivariate-ols-baseline")) {
    warning(
      "mvcm currently computes an unpenalized multivariate OLS baseline; variance components are not estimated.",
      call. = FALSE
    )
  }

  # 3. Reshape flat coefficient arrays back to an R matrix structural block
  coef_matrix <- matrix(
    fit_output$coefficients_matrix_flat,
    nrow = nc,
    ncol = n_resp
  )
  rownames(coef_matrix) <- colnames(X)
  colnames(coef_matrix) <- colnames(Y_matrix)

  fit_output$coefficients_matrix_flat <- NULL
  fit_output$coefficients <- coef_matrix
  class(fit_output) <- "mrvcr_mvcm"
  return(fit_output)
}
