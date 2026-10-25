# Optional EM/MM comparison using the same exact Gaussian ML/REML objective.
# Run from the package root after devtools::load_all():
# source("bench/compare-vcm-em-mm.R")
# The two methods may have different iteration counts. MM is not guaranteed
# to be faster than EM for every model or kernel structure.
set.seed(1553)
n <- 30L
group <- factor(rep(seq_len(6L), each = 5L))
x <- stats::rnorm(n)
d <- data.frame(x = x, group = group)
d$y <- 0.8 + 0.7*x + c(-1.2, 1.3, -.7, 1.4, -1.0, .8)[as.integer(group)] +
  stats::rnorm(n, sd = .23)
Z <- stats::model.matrix(~ 0 + group, d)
kernels <- list(random = tcrossprod(Z), residual = diag(n))
for (pet in c("MLE", "REML")) {
  fits <- list()
  times <- list()
  for (method in c("EM", "MM")) {
    t <- system.time({
      fits[[method]] <- mrvcr::vcm(y ~ x, d, kernels,
        pet = pet, method = method, theta_init = c(.5, .5),
        tol = 1e-9, max_iter = 250L)
    })
    times[[method]] <- unname(t[["elapsed"]])
    z <- fits[[method]]
    cat(sprintf("%-4s %-2s iter=%-4d converged=%-5s logLik=%13.7f elapsed=%.4fs\n",
                pet, method, z$iterations, z$converged,
                z$log_likelihood, times[[method]]))
    stopifnot(all(diff(z$loglik_history) >= -1e-8))
  }
  cat("variance-component differences:",
      max(abs(fits$EM$variance_components - fits$MM$variance_components)), "\n")
}
