#' Variance Component Model with variance kernels V_i = Z_i %*% Z_i^T
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

  # 3. Transmit arrays directly to the compiled rust endpoint
  fit_output <- .Call(
    wrap__vcm_zzt_fit_rust,
    as.numeric(Y),
    as.numeric(X),
    as.numeric(Z),
    as.integer(nrow(X)),
    as.integer(ncol(X)),
    as.integer(ncol(Z)),
    as.character(pet),
    as.character(method)
  )

  # 4. Reconstruct return layout mapping
  fit_output$coefficients <- setNames(fit_output$coefficients, colnames(X))
  class(fit_output) <- "mrvcr_vcm"
  return(fit_output)
}
