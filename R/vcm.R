#' Variance Component Model with variance kernels V_i = Z_i %*% Z_i^T
#' @param formula Fixed-effects regression formula.
#' @param random_formula One-sided formula defining a random-effects design matrix.
#' @param data A data frame or environment containing model variables.
#' @param pet Estimation criterion (`"MLE"` or `"REML"`).
#' @param method Numerical method, currently only the incomplete `"MM"` prototype.
#'
#' @export
vcm_zzt <- function(formula, random_formula, data = parent.frame(), pet = "REML", method = "MM") {
  cl <- match.call()

  # 1. Process standard fixed effects elements
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data"), names(mf), 0L)
  mf_fixed <- mf[c(1L, m)]
  mf_fixed[[1L]] <- quote(stats::model.frame)
  mf_fixed <- eval(mf_fixed, parent.frame())

  X <- stats::model.matrix(attr(mf_fixed, "terms"), mf_fixed)
  Y <- stats::model.response(mf_fixed)

  # 2. Extract random design matrix Z from dedicated sub-formula argument
  # Expects formatting like: random_formula = ~ 0 + factor_variable
  mf_random <- cl
  m_r <- match(c("random_formula", "data"), names(mf_random), 0L)
  mf_random <- mf_random[c(1L, m_r)]
  names(mf_random)[names(mf_random) == "random_formula"] <- "formula"
  mf_random[[1L]] <- quote(stats::model.frame)
  mf_random <- eval(mf_random, parent.frame())

  Z <- stats::model.matrix(attr(mf_random, "terms"), mf_random)

  # Pass matrices to the current 5-argument extendr wrapper. Do not flatten.
  fit_output <- vcm_zzt_fit_rust(
    Y, X, Z, as.character(pet), as.character(method)
  )

  # The current Rust MM branch uses placeholder variance updates. Never
  # expose the output as a valid fitted variance-component model.
  if (!isTRUE(fit_output$converged)) {
    stop(
      "vcm_zzt is currently a prototype: ML/REML MM variance-component updates are not implemented.",
      call. = FALSE
    )
  }

  # 4. Reconstruct return layout mapping
  fit_output$coefficients <- stats::setNames(fit_output$coefficients, colnames(X))
  class(fit_output) <- "mrvcr_vcm"
  return(fit_output)
}
