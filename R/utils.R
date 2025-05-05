# internal argument‐checking helper
.check_mrvcr_args <- function(Y, X, V) {
  if (!is.matrix(Y)) stop("Y must be a numeric matrix.")
  if (!is.null(X) && (!is.matrix(X) || nrow(X)!=nrow(Y)))
    stop("X must be NULL or a matrix with same nrow as Y.")
  if (!is.list(V) || any(sapply(V, function(M) !is.matrix(M))))
    stop("V must be a list of matrices.")
}
