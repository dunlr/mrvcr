# Sparse CSC warm-start path vs repeated independent sparse fits.
# Run from package root: source("bench/compare-lmnet-sparse-path.R")
# End-to-end comparison; below timer resolution is not a zero runtime.
if (!requireNamespace("Matrix", quietly = TRUE)) stop("Install Matrix first.")
set.seed(4291)
n <- 3000L
p <- 250L
i <- sample.int(n, 36000L, replace = TRUE)
j <- sample.int(p, 36000L, replace = TRUE)
x <- Matrix::sparseMatrix(i = i, j = j, x = rnorm(length(i)),
                          dims = c(n, p),
                          dimnames = list(NULL, paste0("x", seq_len(p))))
x <- methods::as(x, "dgCMatrix")
y <- 0.7 + as.vector(x %*% c(1.4, -.8, rep(0, p - 2L))) + rnorm(n, sd = .4)
lambda <- exp(seq(log(.8), log(.03), length.out = 20L))
alpha <- .4
warm_time <- system.time({
  warm <- mrvcr::lmnet_sparse(x, y, pet = "ElasticNet",
                              lambda = lambda, alpha = alpha, tol = 1e-9)
})
cold_time <- system.time({
  cold <- lapply(lambda, function(s) mrvcr::lmnet_sparse(
    x, y, pet = "ElasticNet", lambda = s, alpha = alpha, tol = 1e-9
  ))
})
stopifnot(all(warm$converged), all(vapply(cold, `[[`, logical(1), "converged")))
reference <- do.call(cbind, lapply(cold, function(z) unname(stats::coef(z))))
cat(sprintf("Warm sparse path elapsed: %.4f s\n", warm_time[["elapsed"]]))
cat(sprintf("Separate sparse fits elapsed: %.4f s\n", cold_time[["elapsed"]]))
cat(sprintf("Max absolute coefficient difference: %.4g\n",
            max(abs(unname(stats::coef(warm)) - reference))))
cat(sprintf("Max coordinate sweeps per lambda: %d\n", max(warm$iterations)))
cat(sprintf("CSC input size: %.3f MiB\n", as.numeric(object.size(x))/2^20))
