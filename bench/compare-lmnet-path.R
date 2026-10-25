# Optional performance and accuracy diagnostic for warm-start coefficient paths.
# Run from package root: source("bench/compare-lmnet-path.R")
# This measures the *end-to-end R interface*; use reproducible profiling before
# drawing any broad performance conclusions.
set.seed(314159)
n <- 2500L; p <- 25L
x <- matrix(stats::rnorm(n*p), n, p)
colnames(x) <- paste0("x", seq_len(p))
y <- 1.3 + drop(x %*% c(1.4, -.8, rep.int(0, p-2))) +
  stats::rnorm(n, sd = .5)
d <- as.data.frame(x)
d$y <- y
f <- stats::reformulate(colnames(x), response = "y")
ls <- exp(seq(log(1.1), log(.02), length.out = 24L))
alpha <- .45
warm_time <- system.time({
  warm <- mrvcr::lmnet(f, data=d, pet="ElasticNet", alpha=alpha,
                       lambda=ls, tol=1e-8)
})
cold_time <- system.time({
  cold <- lapply(ls, function(lam) {
    mrvcr::lmnet(f, data=d, pet="ElasticNet", alpha=alpha,
                 lambda=lam, tol=1e-8)
  })
})
stopifnot(all(warm$converged), all(vapply(cold, function(z) z$converged, logical(1))))
ref <- do.call(cbind, lapply(cold, function(z) unname(stats::coef(z))))
cat("Warm-start elapsed seconds:", unname(warm_time["elapsed"]), "\n")
cat("Cold-start elapsed seconds:", unname(cold_time["elapsed"]), "\n")
cat("Maximum coefficient difference:", max(abs(unname(stats::coef(warm)) - ref)), "\n")
cat("Warm-start max sweeps/lambda:", max(warm$iterations), "\n")
