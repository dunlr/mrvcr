#' Fit a multivariate‐response variance‐component model via MM or EM
#'
#' @param Y numeric matrix (n × d) of responses
#' @param X numeric matrix (n × p) of fixed‐effects design (NULL for no fixed effects)
#' @param V list of n × n kernel matrices (length m)
#' @param reml logical; use REML (default FALSE)
#' @param algo character; “MM” or “EM” (default “MM”)
#' @param tol numeric; convergence tolerance (default 1e-8)
#' @param max_iter integer; maximum MM/EM iterations (default 1000)
#' @return A list with elements:
#'   * B: p × d matrix of fixed‐effect estimates
#'   * Sigma: list of d × d variance‐component matrices
#'   * loglik: numeric vector of log‐likelihoods per iteration
#'   * se: list with standard errors for B and Σ
#' @export
fit_mrvc <- function(Y, X = NULL, V, reml = FALSE,
                     algo = c("MM", "EM"), tol = 1e-8, max_iter = 1000) {
  algo <- match.arg(algo)
  # input checks
  if (!is.matrix(Y)) stop("Y must be a matrix")
  n <- nrow(Y); d <- ncol(Y)
  if (is.null(X)) {
    p <- 0; Xmat <- matrix(0, n, 0)
  } else {
    if (!is.matrix(X) || nrow(X) != n) stop("X must be n×p matrix matching Y")
    p <- ncol(X); Xmat <- X
  }
  m <- length(V)
  Vlist <- lapply(V, function(M) {
    if (!is.matrix(M) || nrow(M) != n || ncol(M) != n)
      stop("Each element of V must be an n×n matrix")
    M
  })
  # call C++ backend
  res <- .Call(
    "_mrvcr_rcpp_fit_mrvc",
    as.numeric(Y), n, d,
    if (p>0) as.numeric(Xmat) else numeric(0), p,
    Vlist, as.logical(reml), algo, tol, as.integer(max_iter)
  )
  B     <- matrix(res$B, nrow = p, ncol = d)
  Sigma <- lapply(res$Sigma, function(v) matrix(v, nrow = d, ncol = d))
  loglik <- res$loglik
  se    <- res$se
  list(B = B, Sigma = Sigma, loglik = loglik, se = se)
}
