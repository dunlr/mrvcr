# Optional diagnostic: compare Gaussian EM/MM using identical starting values
# and both the legacy stopping criterion and the scaled-score KKT criterion.
# Run from the package root after devtools::load_all().
set.seed(511)
n <- 40L
g <- rep(seq_len(8L), each = 5L)
d <- data.frame(x = stats::rnorm(n))
d$y <- 0.5 + 0.8*d$x + stats::rnorm(8L)[g] + stats::rnorm(n, sd=.3)
Z <- stats::model.matrix(~ 0 + factor(g))
K <- list(random = tcrossprod(Z), residual = diag(n))
for (strategy in c("equal", "trace", "moments")) {
  # Use initial values from an exploratory fit but *not* its endpoint.
  first <- suppressWarnings(mrvcr::vcm(y ~ x, d, K, init = strategy,
                                        method = "EM", max_iter = 1L))
  init <- first$theta_initial
  cat("\nInitialization:", strategy, "| theta0:",
      paste(signif(init, 4L), collapse = ", "), "\n")
  for (pet in c("MLE", "REML")) {
    for (method in c("EM", "MM")) {
      elapsed <- system.time({
        fit <- suppressWarnings(mrvcr::vcm(y ~ x, d, K,
          theta_init = init, pet = pet, method = method,
          tol = 1e-7, kkt_tol = 1e-5, stop_rule = "both",
          max_iter = 500L))
      })[["elapsed"]]
      cat(sprintf("  %-4s %-2s  iter=%-4d converged=%-5s LL=%10.5f KKT=%9.3g time=%.3fs\n",
                  pet, method, fit$iterations, fit$converged,
                  fit$log_likelihood, fit$kkt_violation_scaled, elapsed))
      stopifnot(all(diff(fit$loglik_history) >= -1e-7))
      if (fit$converged) stopifnot(fit$converged_parameters,
                                 fit$converged_kkt)
    }
  }
}
# End-to-end R interface timing; replicate under controlled conditions before
# making speed claims. MM and EM may differ in iteration count and runtime.
