# Optional end-to-end diagnostic for Matrix::sparse.model.matrix() dispatch.
# From package root: source("bench/compare-lmnet-sparse-formula.R")
if (!requireNamespace("Matrix", quietly = TRUE)) stop("Install Matrix first.")
set.seed(4308)
n <- 3000L
p <- 200L
d <- data.frame(group = factor(sample(sprintf("L%03d", seq_len(p)), n, TRUE)),
                z = rnorm(n))
d$y <- .8 + .5 * d$z + as.numeric(d$group) / p + rnorm(n, sd = .3)
f <- y ~ group + z
lambda <- exp(seq(log(.4), log(.03), length.out = 15L))
sparse_t <- system.time({
  s <- mrvcr::lmnet(f, data = d, pet = "ElasticNet", alpha = .4,
                    lambda = lambda, sparse = TRUE, x = TRUE)
})
dense_t <- system.time({
  r <- mrvcr::lmnet(f, data = d, pet = "ElasticNet", alpha = .4,
                    lambda = lambda, x = TRUE)
})
stopifnot(all(s$converged), all(r$converged))
cat(sprintf("Sparse formula path: %.4f s; dense formula path: %.4f s\n",
            sparse_t[["elapsed"]], dense_t[["elapsed"]]))
cat(sprintf("Maximum coefficient difference: %.8g\n",
            max(abs(unname(coef(s)) - unname(coef(r))))))
cat(sprintf("Sparse model matrix: %.3f MiB; dense model matrix: %.3f MiB\n",
            as.numeric(object.size(s[["x", exact = TRUE]]))/2^20,
            as.numeric(object.size(r[["x", exact = TRUE]]))/2^20))
# One trial; timings include R formula processing and should not be generalized.
