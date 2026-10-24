# Diagnostic: Gaussian Ridge/Lasso/ElasticNet objective parity.
# Execute from package root: source("bench/compare-lmnet-glmnet.R")
# Optional script, not executed by R CMD check; requires glmnet.
#
# IMPORTANT: build formulas explicitly from X's column names. Do not use
# y ~ . after adding y_scaled to a data frame: y_scaled would become a
# predictor and invalidate every coefficient comparison.
#
# mrvcr's single-response, unit-weight Gaussian objective is
#   RSS / (2*n) + lambda * [alpha * sum(abs(beta_j)) +
#                           (1-alpha)/2 * sum(beta_j^2)],
# with the intercept unpenalized and X not automatically standardized.

if (!requireNamespace("glmnet", quietly = TRUE)) {
  stop("Install 'glmnet' to run this optional diagnostic.", call. = FALSE)
}

check_beta <- function(b, X, label) {
  expected <- ncol(X) + 1L
  if (!is.numeric(b) || length(b) != expected || any(!is.finite(b))) {
    stop(sprintf("%s: expected %d finite coefficients (intercept plus %d predictors), got %d.",
                 label, expected, ncol(X), length(b)), call. = FALSE)
  }
  invisible(TRUE)
}

first_order_violation <- function(X, y, b, lambda, alpha) {
  check_beta(b, X, "KKT input")
  n <- nrow(X)
  residual <- y - as.vector(b[1L] + X %*% b[-1L])
  gradient <- as.vector(crossprod(X, residual)) / n
  slope <- b[-1L]
  smooth <- gradient - lambda * (1 - alpha) * slope
  cutoff <- lambda * alpha
  subgrad_error <- ifelse(
    slope > 0, abs(smooth - cutoff),
    ifelse(slope < 0, abs(smooth + cutoff), pmax(abs(smooth) - cutoff, 0))
  )
  max(abs(mean(residual)), subgrad_error)
}

gaussian_objective <- function(X, y, b, lambda, alpha) {
  check_beta(b, X, "Objective input")
  residual <- y - as.vector(b[1L] + X %*% b[-1L])
  slope <- b[-1L]
  sum(residual^2) / (2 * nrow(X)) +
    lambda * (alpha * sum(abs(slope)) +
              0.5 * (1 - alpha) * sum(slope^2))
}

numerical_ridge <- function(X, y, lambda) {
  n <- nrow(X)
  design <- cbind(`(Intercept)` = 1, X)
  p <- ncol(design)
  as.vector(solve(crossprod(design) / n +
                    lambda * diag(c(0, rep(1, p - 1L)), nrow = p),
                  crossprod(design, y) / n))
}

glmnet_at_lambda <- function(X, y, lambda, alpha) {
  # Descending lambda path; request a fitted point instead of interpolating.
  fit <- glmnet::glmnet(
    X, y, family = "gaussian", alpha = alpha,
    lambda = c(4 * lambda, 2 * lambda, lambda),
    intercept = TRUE, standardize = FALSE,
    control = list(thresh = 1e-12, maxit = 100000L)
  )
  coef_matrix <- stats::coef(fit, s = lambda)
  expected_names <- c("(Intercept)", colnames(X))
  if (!identical(rownames(coef_matrix), expected_names)) {
    stop("glmnet coefficient names differ from the intended predictors.",
         call. = FALSE)
  }
  b <- as.vector(coef_matrix)
  check_beta(b, X, "glmnet")
  b
}

set.seed(849)
n <- 900L
p <- 15L
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- paste0("x", seq_len(p))
truth <- c(1.5, -0.8, rep(0, p - 2L))
y <- 2 + drop(X %*% truth) + rnorm(n, sd = 0.7)

d <- as.data.frame(X)
d$y <- y
y_center <- mean(y)
y_scale <- sqrt(mean((y - y_center)^2)) # population (1/n) SD
d$y_scaled <- (y - y_center) / y_scale

# ONLY x1, ..., x15 are included, even when auxiliary columns exist in d.
formula_raw <- stats::reformulate(colnames(X), response = "y")
formula_scaled <- stats::reformulate(colnames(X), response = "y_scaled")

lambda <- 0.12
cat(sprintf("Gaussian response scale (1/n): %.8f\n", y_scale))
cat(sprintf("Regression predictors: %d (plus unpenalized intercept)\n\n", p))

for (alpha in c(0, 0.3, 1)) {
  pet <- if (alpha == 0) "Ridge" else if (alpha == 1) "Lasso" else "ElasticNet"

  ours <- mrvcr::lmnet(formula_raw, data = d, pet = pet,
                       lambda = lambda, alpha = alpha, tol = 1e-10)
  ours_names <- names(stats::coef(ours))
  if (!identical(ours_names, c("(Intercept)", colnames(X)))) {
    stop(sprintf("%s: mrvcr used unexpected predictors: %s",
                 pet, paste(ours_names, collapse = ", ")), call. = FALSE)
  }
  b_ours <- unname(as.vector(stats::coef(ours)))
  check_beta(b_ours, X, "mrvcr raw y")
  b_glmnet <- glmnet_at_lambda(X, y, lambda, alpha)

  cat(sprintf("=== %-10s alpha = %.1f | raw y ===\n", pet, alpha))
  cat(sprintf("coefficient gap:                %.8g\n",
              max(abs(b_ours - b_glmnet))))
  cat(sprintf("mrvcr KKT residual:             %.8g\n",
              first_order_violation(X, y, b_ours, lambda, alpha)))
  cat(sprintf("glmnet KKT residual:            %.8g\n",
              first_order_violation(X, y, b_glmnet, lambda, alpha)))
  cat(sprintf("mrvcr objective:                %.10g\n",
              gaussian_objective(X, y, b_ours, lambda, alpha)))
  cat(sprintf("glmnet objective (same scale):  %.10g\n",
              gaussian_objective(X, y, b_glmnet, lambda, alpha)))
  if (!isTRUE(all.equal(ours$objective,
                        gaussian_objective(X, y, b_ours, lambda, alpha),
                        tolerance = 1e-7))) {
    stop(sprintf("%s: Rust-reported objective differs from independently computed objective.", pet),
         call. = FALSE)
  }
  if (alpha == 0) {
    b_exact <- numerical_ridge(X, y, lambda)
    cat(sprintf("mrvcr vs analytic ridge:        %.8g\n",
                max(abs(b_ours - b_exact))))
    cat(sprintf("glmnet vs analytic ridge:       %.8g\n",
                max(abs(b_glmnet - b_exact))))
  }

  ours_scaled <- mrvcr::lmnet(formula_scaled, data = d, pet = pet,
                              lambda = lambda, alpha = alpha, tol = 1e-10)
  scaled_names <- names(stats::coef(ours_scaled))
  if (!identical(scaled_names, c("(Intercept)", colnames(X)))) {
    stop(sprintf("%s: mrvcr scaled-response fit used unexpected predictors.", pet),
         call. = FALSE)
  }
  b_ours_scaled <- unname(as.vector(stats::coef(ours_scaled)))
  check_beta(b_ours_scaled, X, "mrvcr standardized y")
  b_glmnet_scaled <- glmnet_at_lambda(X, d$y_scaled, lambda, alpha)
  cat(sprintf("standardized-y coefficient gap: %.8g\n\n",
              max(abs(b_ours_scaled - b_glmnet_scaled))))
}

cat("Interpret results using the objective and KKT checks, not the\n")
cat("coefficient gap alone. glmnet documents internal response scaling,\n")
cat("but this diagnostic does not assume it explains any remaining gap.\n")
