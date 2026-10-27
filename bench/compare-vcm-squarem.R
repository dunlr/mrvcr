# Iteration 5E: compare base EM/MM versus safeguarded SQUAREM.
# Run from the package root after devtools::load_all().
# Times include data validation and spectral decomposition. Repeated timings
# are recommended; low durations may be beneath system.time resolution.
set.seed(20261025)
n <- 120L
ng <- 24L
group <- factor(rep(seq_len(ng), each = n / ng))
d <- data.frame(x = stats::rnorm(n), group = group)
d$y <- 0.7 + 0.4 * d$x + stats::rnorm(ng, sd = 0.8)[as.integer(group)] +
  stats::rnorm(n, sd = 0.3)
Z <- stats::model.matrix(~ 0 + group, d)
K <- list(random = tcrossprod(Z), residual = diag(n))
for (solver in c("spectral", "dense")) {
  for (pet in c("MLE", "REML")) for (method in c("EM", "MM")) {
    fits <- list()
    for (acceleration in c("none", "SQUAREM")) {
      elapsed <- system.time({
        fits[[acceleration]] <- suppressWarnings(mrvcr::vcm(
          y ~ x, d, K, pet = pet, method = method, solver = solver,
          acceleration = acceleration, theta_init = c(0.6, 0.2),
          tol = 1e-7, kkt_tol = 1e-5, stop_rule = "both",
          max_iter = 400L
        ))
      })[["elapsed"]]
      fit <- fits[[acceleration]]
      cat(sprintf(
        "%8s %-4s %-2s %-6s cycles=%3d F=%3d LL=%3d accepted=%2d KKT=%8.2g converged=%-5s sec=%.3f\n",
        solver, pet, method, acceleration, fit$iterations,
        fit$fixed_point_evaluations, fit$likelihood_evaluations,
        fit$squarem_accepted, fit$kkt_violation_scaled, fit$converged,
        unname(elapsed)))
      stopifnot(all(diff(fit$loglik_history) >= -1e-7))
    }
    cat("  loglik gap =",
        abs(fits$none$log_likelihood - fits$SQUAREM$log_likelihood),
        "component gap =",
        max(abs(fits$none$variance_components - fits$SQUAREM$variance_components)),
        "\n")
  }
}
