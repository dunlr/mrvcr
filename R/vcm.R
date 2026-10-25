#' Gaussian Variance Component Model via Exact ML/REML, EM and MM (Experimental)
#'
#' Fits `Y ~ N(X beta, Sigma)` where `Sigma = sum_k theta[k] * kernels[[k]]`.
#' Covariance kernels must be known symmetric positive-semidefinite `n x n`
#' matrices in exactly the same **row order** as the model frame. This dense
#' initial implementation validates the kernels using symmetric eigenvalues,
#' then evaluates the Gaussian (restricted) log likelihood and fits the
#' nonnegative variance components using either latent-component EM or the
#' multiplicative minorization-maximization (MM) algorithm.
#'
#' `pet = "MLE"` profiles out beta in the ordinary likelihood;
#' `pet = "REML"` uses the restricted likelihood, including the
#' `log det(X' Sigma^-1 X)` term. REML likelihood values depend on the
#' chosen fixed-effects design basis and cannot freely be compared across
#' different fixed-effect formulae. No Laplace or quadrature approximation is
#' involved in this Gaussian model.
#'
#' Let `q[k] = (y-X beta_GLS)' Sigma^-1 K[k] Sigma^-1 (y-X beta_GLS)`.
#' Let `t[k] = tr(P K[k])`, where `P = Sigma^-1` for profiled ML,
#' and `P = Sigma^-1 - Sigma^-1 X (X' Sigma^-1 X)^-1 X' Sigma^-1`
#' for REML. The EM update is `theta[k] + theta[k]^2 * (q[k]-t[k]) /
#' rank(K[k])`; the distinct MM update is `theta[k] * sqrt(q[k]/t[k])`.
#' The MM update is derived from a minorization of the likelihood, not
#' from a numerical integration approximation. Initial components must be
#' strictly positive; estimates may approach the zero boundary. Identifiability
#' is not guaranteed for redundant kernels. This implementation is dense and
#' intended for small or moderate models; its computational cost is cubic in
#' n per EM or MM step.
#'
#' @param formula Fixed-effects formula, including response; numeric univariate y.
#' @param data Data frame/environment used to construct the model frame.
#' @param kernels Nonempty named or unnamed list of known n-by-n symmetric
#'   positive-semidefinite covariance kernels in model-frame observation order.
#'   Include a residual kernel such as `diag(n)` to ensure positive-definite
#'   covariance whenever appropriate. Missing observations are rejected.
#' @param pet `"MLE"` (alias `"ML"`) or `"REML"`.
#' @param method `"EM"` (latent Gaussian expectation-maximization) or `"MM"`
#'   (multiplicative minorization-maximization).
#' @param theta_init Optional strictly positive numeric starting values, one
#'   per kernel. Named vectors are matched by kernel name; a previous
#'   `mrvcr_vcm_em` fit supplies a warm start. If omitted, `init` determines
#'   the initializer.
#' @param init Starting-value strategy when `theta_init = NULL`:
#'   `"equal"` (legacy default), `"trace"` (scale components by their
#'   covariance contributions), `"moments"` (nonnegative residual moment
#'   regression), or `"random"` (randomly perturb trace-scaled starts).
#'   The moment estimator is for **initialization only**, not a separate PET.
#' @param stop_rule `"parameter"` (backward-compatible default), `"kkt"`,
#'   or `"both"`. KKT stopping uses a scale-adjusted score residual.
#' @param kkt_tol Scaled KKT tolerance, default `sqrt(tol)`. Applies when
#'   `stop_rule` includes KKT. Always reports diagnostics regardless.
#' @param boundary_tol Nonnegative relative threshold for classifying an
#'   estimated variance component as approximately zero in KKT checks.
#'   This does not project coefficients or modify the EM/MM updates.
#' @param max_iter Positive integer number of optimizer iterations.
#' @param tol Positive relative parameter-change stopping tolerance.
#' @param contrasts Optional fixed-effects contrasts passed to `model.matrix`.
#' @param model Whether to store the model frame.
#' @param x Whether to retain the dense fixed-effects model matrix.
#' @return An experimental `mrvcr_vcm_em` object (legacy class name retained
#'   for S3 compatibility even for MM fits) with GLS coefficients,
#'   fitted values, residuals, component estimates, rank information,
#'   log likelihood, exact profile score, raw and scaled KKT violations,
#'   approximate-boundary flags, initialization metadata, stopping diagnostics,
#'   and the likelihood trajectory. The scale-adjusted KKT diagnostic uses the per-component divisor
#'   `1 + (abs(q[k]) + abs(t[k])) / 2`; its tolerance is not an unscaled score
#'   tolerance. Parameter-only convergence does not certify KKT stationarity.
#' @export
vcm <- function(formula, data = parent.frame(), kernels,
                pet = "REML", method = "EM", theta_init = NULL,
                max_iter = 500L, tol = 1e-7, contrasts = NULL,
                model = TRUE, x = FALSE,
                init = "equal", stop_rule = "parameter", kkt_tol = sqrt(tol),
                boundary_tol = 1e-8) {
  if (!inherits(formula, "formula")) stop("'formula' must be a formula.", call. = FALSE)
  if (missing(kernels) || !is.list(kernels) || length(kernels) < 1L) {
    stop("'kernels' must be a nonempty list of covariance matrices.", call. = FALSE)
  }
  if (!is.character(pet) || length(pet) != 1L || is.na(pet) ||
      !toupper(pet) %in% c("ML", "MLE", "REML")) {
    stop("'pet' must be 'MLE' or 'REML'.", call. = FALSE)
  }
  pet <- if (toupper(pet) %in% c("ML", "MLE")) "MLE" else "REML"
  if (!is.character(method) || length(method) != 1L || is.na(method) ||
      !toupper(method) %in% c("EM", "MM")) {
    stop("'method' must be 'EM' or 'MM'.", call. = FALSE)
  }
  method <- toupper(method)
  if (!is.numeric(tol) || length(tol) != 1L || !is.finite(tol) || tol <= 0) {
    stop("'tol' must be positive and finite.", call. = FALSE)
  }
  if (!is.character(init) || length(init) != 1L || is.na(init) ||
      !init %in% c("equal", "trace", "moments", "random")) {
    stop("'init' must be 'equal', 'trace', 'moments', or 'random'.", call. = FALSE)
  }
  if (!is.character(stop_rule) || length(stop_rule) != 1L || is.na(stop_rule) ||
      !stop_rule %in% c("parameter", "kkt", "both")) {
    stop("'stop_rule' must be 'parameter', 'kkt', or 'both'.", call. = FALSE)
  }
  if (!is.numeric(kkt_tol) || length(kkt_tol) != 1L ||
      !is.finite(kkt_tol) || kkt_tol <= 0) {
    stop("'kkt_tol' must be positive and finite.", call. = FALSE)
  }
  if (!is.numeric(boundary_tol) || length(boundary_tol) != 1L ||
      !is.finite(boundary_tol) || boundary_tol < 0) {
    stop("'boundary_tol' must be nonnegative and finite.", call. = FALSE)
  }
  if (!is.numeric(max_iter) || length(max_iter) != 1L ||
      !is.finite(max_iter) || max_iter < 1L || max_iter != floor(max_iter) ||
      max_iter > .Machine$integer.max) {
    stop("'max_iter' must be a positive integer.", call. = FALSE)
  }
  if (!is.logical(model) || length(model) != 1L || is.na(model) ||
      !is.logical(x) || length(x) != 1L || is.na(x)) {
    stop("'model' and 'x' must each be TRUE or FALSE.", call. = FALSE)
  }
  mf <- stats::model.frame(formula = formula, data = data,
                           na.action = stats::na.fail, drop.unused.levels = TRUE)
  response <- stats::model.response(mf)
  if (!is.numeric(response) || !is.null(dim(response)) ||
      any(!is.finite(response))) {
    stop("VCM currently requires a finite numeric univariate response.", call. = FALSE)
  }
  if (!is.null(stats::model.offset(mf))) {
    stop("Offsets in the fixed-effects formula are not supported yet.", call. = FALSE)
  }
  terms <- attr(mf, "terms")
  X <- stats::model.matrix(terms, data = mf, contrasts.arg = contrasts)
  storage.mode(X) <- "double"
  n <- NROW(response)
  p <- NCOL(X)
  if (n < 2L || p < 1L || (pet == "REML" && p >= n)) {
    stop("Need n > p >= 1 for REML and n >= 2 for ML.", call. = FALSE)
  }
  if (qr(X)$rank != p) {
    stop("The fixed-effects model matrix must have full column rank.", call. = FALSE)
  }
  m <- length(kernels)
  kernel_names <- names(kernels)
  if (is.null(kernel_names)) kernel_names <- rep.int("", m)
  missing_names <- is.na(kernel_names) | !nzchar(kernel_names)
  kernel_names[missing_names] <- paste0("V", which(missing_names))
  if (anyDuplicated(kernel_names)) stop("Kernel names must be unique.", call. = FALSE)
  # Rank and positive-semidefiniteness validation is explicit here. This
  # avoids accidentally interpreting an indefinite kernel as a covariance.
  rank <- integer(m)
  K <- array(0, dim = c(n, n, m))
  for (j in seq_len(m)) {
    kj <- kernels[[j]]
    if (!is.matrix(kj) || !is.numeric(kj) ||
        !identical(dim(kj), c(n, n)) || any(!is.finite(kj))) {
      stop(sprintf("Kernel '%s' must be a finite numeric %d-by-%d matrix.",
                   kernel_names[j], n, n), call. = FALSE)
    }
    kj <- matrix(as.numeric(kj), nrow = n, ncol = n)
    scale <- max(1, abs(kj))
    sym_tol <- 1e-10 * scale
    if (max(abs(kj - t(kj))) > sym_tol) {
      stop(sprintf("Kernel '%s' is not symmetric.", kernel_names[j]), call. = FALSE)
    }
    kj <- (kj + t(kj)) / 2
    eigenvalues <- eigen(kj, symmetric = TRUE, only.values = TRUE)$values
    rank_tol <- max(1, max(abs(eigenvalues))) * n * sqrt(.Machine$double.eps)
    if (min(eigenvalues) < -rank_tol) {
      stop(sprintf("Kernel '%s' is not positive semidefinite.", kernel_names[j]),
           call. = FALSE)
    }
    rank[j] <- sum(eigenvalues > rank_tol)
    if (rank[j] == 0L) {
      stop(sprintf("Kernel '%s' has numerical rank zero.", kernel_names[j]), call. = FALSE)
    }
    K[, , j] <- kj
  }
  initial_method <- if (is.null(theta_init)) init else "user"
  if (inherits(theta_init, "mrvcr_vcm_em")) {
    theta_init <- theta_init$variance_components
    if (!is.numeric(theta_init) || any(!is.finite(theta_init)) ||
        any(theta_init < 0)) {
      stop("Warm-start fit must contain finite nonnegative variance components.", call. = FALSE)
    }
    # A zero estimate is a legitimate boundary solution. EM/MM require
    # strictly positive starts if that component may later re-enter.
    theta_init <- pmax(theta_init, 1e-8 * max(1, theta_init))
    initial_method <- "warm_start"
  }
  if (is.null(theta_init)) {
    theta_init <- .vcm_initialize(response, X, K, init)
  }
  if (!is.numeric(theta_init) || length(theta_init) != m ||
      any(!is.finite(theta_init)) || any(theta_init <= 0)) {
    stop("'theta_init' must contain one strictly positive finite value per kernel.",
         call. = FALSE)
  }
  if (!is.null(names(theta_init))) {
    if (anyNA(names(theta_init)) || anyDuplicated(names(theta_init)) ||
        !setequal(names(theta_init), kernel_names)) {
      stop("Named 'theta_init' must contain exactly the fitted kernel names.",
           call. = FALSE)
    }
    theta_init <- theta_init[kernel_names]
  }
  initial_theta <- stats::setNames(as.numeric(theta_init), kernel_names)
  result <- vcm_em_fit_rust(
    as.numeric(response), X, K, as.integer(rank), as.numeric(theta_init),
    pet, method, as.integer(max_iter), as.numeric(tol),
    stop_rule, as.numeric(kkt_tol), as.numeric(boundary_tol)
  )
  result$coefficients <- stats::setNames(as.numeric(result$coefficients), colnames(X))
  result$variance_components <- stats::setNames(as.numeric(result$variance_components), kernel_names)
  result$fitted.values <- stats::setNames(as.numeric(result$fitted_values), rownames(mf))
  result$residuals <- stats::setNames(as.numeric(result$residuals), rownames(mf))
  result$kernel_ranks <- stats::setNames(as.integer(rank), kernel_names)
  result$score <- stats::setNames(as.numeric(result$score), kernel_names)
  result$boundary_components <- stats::setNames(as.logical(result$boundary_components), kernel_names)
  result$theta_initial <- initial_theta
  result$initialization <- initial_method
  result$call <- match.call()
  result$terms <- terms
  result$xlevels <- stats::.getXlevels(terms, mf)
  result$contrasts <- attr(X, "contrasts")
  result$nobs <- n
  result$ncoef <- p
  if (model) result$model <- mf
  if (x) result$x <- X
  class(result) <- "mrvcr_vcm_em"
  if (!result$converged) {
    warning(sprintf("VCM %s reached max_iter without satisfying stop_rule = '%s'; examine kkt_violation_scaled and loglik_history.",
                    method, stop_rule),
            call. = FALSE)
  }
  result
}

#' @export
coef.mrvcr_vcm_em <- function(object, ...) object$coefficients

#' @export
fitted.mrvcr_vcm_em <- function(object, ...) object$fitted.values

#' @export
residuals.mrvcr_vcm_em <- function(object, ...) object$residuals

#' @export
predict.mrvcr_vcm_em <- function(object, newdata = NULL, ...) {
  if (is.null(newdata)) return(stats::fitted(object))
  tt <- stats::delete.response(object$terms)
  mf <- stats::model.frame(tt, data = newdata, na.action = stats::na.fail,
                           xlev = object$xlevels)
  xm <- stats::model.matrix(tt, data = mf, contrasts.arg = object$contrasts)
  if (!identical(colnames(xm), names(object$coefficients))) {
    stop("Prediction matrix columns do not match the fitted fixed-effects columns.", call. = FALSE)
  }
  stats::setNames(as.vector(xm %*% object$coefficients), rownames(mf))
}

#' @export
print.mrvcr_vcm_em <- function(x, ...) {
  cat("Gaussian variance component model (", x$pet, "/", x$method, ")\n", sep = "")
  cat("Log likelihood: ", format(x$log_likelihood, digits = 8),
      "; iterations: ", x$iterations,
      "; converged: ", x$converged, "\n", sep = "")
  cat("KKT violation (scaled): ", format(x$kkt_violation_scaled, digits = 5),
      "; stopping rule: ", x$stop_rule, "\n", sep = "")
  cat("Variance components:\n")
  print(x$variance_components)
  invisible(x)
}

# OLS residuals and residualized kernels give linear moment equations:
# E[r' (M K_k M) r] = sum_j theta_j tr[(M K_k M)(M K_j M)].
# Solving the associated nonnegative quadratic problem provides a starting
# value, not a new likelihood criterion. All outputs are kept strictly
# positive for the existing EM/MM kernels.
.vcm_initialize <- function(y, X, K, strategy) {
  m <- dim(K)[3L]
  if (strategy == "equal") {
    return(rep.int(max(stats::var(y) / m, 1e-4), m))
  }
  n <- length(y)
  # A QR basis avoids explicit inversion of X'X in the projector.
  basis <- qr.Q(qr(X))
  M <- diag(n) - tcrossprod(basis)
  r <- as.vector(M %*% y)
  df <- n - ncol(X)
  rss_per_df <- if (df > 0L) sum(r^2) / df else stats::var(y)
  base_variance <- max(rss_per_df, 1e-4)
  projected_trace <- vapply(seq_len(m), function(k) {
    sum(M * K[, , k]) / max(df, 1L)
  }, numeric(1))
  raw_trace <- vapply(seq_len(m), function(k) {
    sum(diag(K[, , k])) / n
  }, numeric(1))
  # Kernels confined to col(X) have no residual moments. Use an
  # unprojected trace as a fallback for initialization only.
  scales <- ifelse(projected_trace > 1e-12 * pmax(1, raw_trace),
                   projected_trace, raw_trace)
  if (any(!is.finite(scales)) || any(scales <= 0)) {
    stop("Cannot scale VCM initial values: a covariance kernel has negligible trace.",
         call. = FALSE)
  }
  initial <- pmax(base_variance / (m * scales), 1e-8)
  if (strategy == "trace") return(initial)
  if (strategy == "random") {
    return(initial * exp(stats::runif(m, log(0.5), log(2))))
  }
  # Projected quadratic moment regression. Forming M K M is dense; this
  # initial implementation is intended for moderate n, as is the solver.
  adjusted <- lapply(seq_len(m), function(k) M %*% K[, , k] %*% M)
  target <- vapply(adjusted, function(a) {
    sum(r * as.vector(a %*% r))
  }, numeric(1))
  gram <- matrix(0.0, nrow = m, ncol = m)
  for (a in seq_len(m)) for (b in seq_len(a)) {
    gram[a, b] <- gram[b, a] <- sum(adjusted[[a]] * adjusted[[b]])
  }
  objective <- function(theta) {
    0.5 * sum(theta * as.vector(gram %*% theta)) - sum(target * theta)
  }
  gradient <- function(theta) as.vector(gram %*% theta - target)
  candidate <- tryCatch(stats::optim(
    initial, objective, gr = gradient, method = "L-BFGS-B",
    lower = rep.int(0, m), control = list(maxit = 1000L)
  ), error = function(e) NULL)
  if (is.null(candidate) || candidate$convergence != 0L ||
      any(!is.finite(candidate$par))) {
    warning("Moment initializer did not converge; using trace-scaled values.",
            call. = FALSE)
    return(initial)
  }
  pmax(candidate$par, 1e-8 * max(1, max(candidate$par)))
}
