#' Fit a ZZT variance-component model via Gaussian EM or MM
#'
#' Compatibility interface for models whose random-effect covariance kernel is
#' `Z %*% t(Z)`. This adapter calls [vcm()] with covariance kernels
#' `list(random = tcrossprod(Z), residual = diag(n))` and therefore fits two
#' variance components, one for random effects and one for residual noise.
#'
#' The previous `vcm_zzt()` implementation was an unfinished MM prototype
#' with invalid placeholder updates. The default `method = "MM"` now calls
#' the actual minorization-maximization optimizer of [vcm()], using the
#' same exact Gaussian marginal or restricted likelihood as `method = "EM"`.
#' Existing EM calls continue to use the unchanged EM optimizer.
#'
#' The two model frames must refer to exactly the same observations, in exactly
#' the same order, without missing values. The random-effects formula must be
#' one-sided (for example, `~ 0 + group`). This is a dense compatibility
#' interface: the resulting `Z Z^T` kernel is materialized as a dense matrix.
#'
#' @param formula Two-sided fixed-effects model formula, including the response.
#' @param random_formula One-sided random-effects design formula, such as
#'   `~ 0 + group`.
#' @param data Data used in the fixed and random model frames.
#' @param pet `"MLE"` or `"REML"`.
#' @param method `"MM"` (default) or `"EM"`; both use the validated VCM engine.
#' @param theta_init Optional length-two positive initial variance vector
#'   (random effect and residual, respectively).
#' @param max_iter Maximum EM/MM iterations.
#' @param tol Positive convergence tolerance.
#' @param model Whether to retain the fixed-effects model frame.
#' @param x Whether to retain the fixed-effects design matrix.
#' @return An `mrvcr_vcm_em` object returned by [vcm()].
#' @seealso [vcm()]
#' @export
vcm_zzt <- function(formula, random_formula, data = parent.frame(),
                    pet = "REML", method = "MM", theta_init = NULL,
                    max_iter = 500L, tol = 1e-7,
                    model = TRUE, x = FALSE) {
  if (!is.character(method) || length(method) != 1L || is.na(method) ||
      !toupper(method) %in% c("EM", "MM")) {
    stop("'method' must be 'EM' or 'MM'.", call. = FALSE)
  }
  method <- toupper(method)
  if (!inherits(formula, "formula") || length(formula) != 3L) {
    stop("'formula' must be a two-sided formula with a response.", call. = FALSE)
  }
  if (!inherits(random_formula, "formula") || length(random_formula) != 2L) {
    stop("'random_formula' must be a one-sided formula, e.g. ~ 0 + group.",
         call. = FALSE)
  }
  fixed_mf <- stats::model.frame(formula, data = data,
                                  na.action = stats::na.fail,
                                  drop.unused.levels = TRUE)
  random_mf <- stats::model.frame(random_formula, data = data,
                                   na.action = stats::na.fail,
                                   drop.unused.levels = TRUE)
  if (!identical(rownames(fixed_mf), rownames(random_mf))) {
    stop("Fixed and random model frames must have identical observations in the same order.",
         call. = FALSE)
  }
  Z <- stats::model.matrix(attr(random_mf, "terms"), data = random_mf)
  if (ncol(Z) < 1L || any(!is.finite(Z))) {
    stop("The random-effects model matrix must contain finite numeric columns.",
         call. = FALSE)
  }
  n <- nrow(Z)
  fit <- vcm(
    formula = formula,
    data = data,
    kernels = list(random = tcrossprod(Z), residual = diag(n)),
    pet = pet,
    method = method,
    theta_init = theta_init,
    max_iter = max_iter,
    tol = tol,
    model = model,
    x = x
  )
  fit$call <- match.call()
  fit
}
