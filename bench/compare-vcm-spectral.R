# Compare EXACT Gaussian ML/REML and EM/MM using identical starts and stops.
# Run after devtools::load_all() from the repository root.
# Includes one-time spectral decomposition, R validations and optimization.
# Timings vary with n, p, m, iteration counts and BLAS configuration.
set.seed(20261025)
n <- 160L
group <- factor(rep(seq_len(32L), each = 5L))
d <- data.frame(x = rnorm(n), group = group)
d$y <- 0.8 + 0.65*d$x + rnorm(32L, sd = 0.85)[as.integer(group)] +
  rnorm(n, sd = 0.3)
Z <- stats::model.matrix(~ 0 + group, d)
kernels <- list(random = tcrossprod(Z), residual = diag(n))
for (pet in c("MLE", "REML")) for (method in c("EM", "MM")) {
  results <- list()
  for (solver in c("dense", "spectral")) {
    tm <- system.time({
      results[[solver]] <- mrvcr::vcm(
        y ~ x, d, kernels, pet = pet, method = method, solver = solver,
        theta_init = c(0.5, 0.25), max_iter = 500L, tol = 1e-7,
        stop_rule = "both", kkt_tol = 1e-5
      )
    })
    r <- results[[solver]]
    cat(sprintf("%-4s %-2s %-8s iter=%-4d converged=%-5s LL=%12.6f KKT=%.3g elapsed=%.3fs\n",
                pet, method, solver, r$iterations, r$converged,
                r$log_likelihood, r$kkt_violation_scaled,
                unname(tm[["elapsed"]])))
    stopifnot(all(diff(r$loglik_history) >= -1e-7))
  }
  cat("max variance-component gap:",
      max(abs(results$dense$variance_components -
                results$spectral$variance_components)), "\n")
  cat("log-likelihood gap:",
      abs(results$dense$log_likelihood - results$spectral$log_likelihood), "\n")
}
