#' Multivariate Response Variance Component model
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

  nr <- as.integer(nrow(X))
  n_resp <- as.integer(ncol(Y_matrix))
  nc <- as.integer(ncol(X))

  # 2. Hand off flattened column-major vectors straight to Rust
  fit_output <- .Call(
    wrap__mvcm_fit_rust,
    as.numeric(Y_matrix),
    as.numeric(X),
    nr,
    n_resp,
    nc,
    as.character(pet)
  )

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
