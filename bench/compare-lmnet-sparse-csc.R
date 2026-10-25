# Optional: source("bench/compare-lmnet-sparse-csc.R") from repository root.
# End-to-end R + Rust timing; not a general performance claim.
if (!requireNamespace("Matrix", quietly = TRUE)) stop("Install Matrix first.")
set.seed(6219)
n <- 3500L
p <- 220L
i <- sample.int(n, 50000L, replace = TRUE)
j <- sample.int(p, 50000L, replace = TRUE)
x <- Matrix::sparseMatrix(i = i, j = j, x = rnorm(length(i)),
                          dims = c(n,p), dimnames = list(NULL,paste0("x",seq_len(p))))
x <- methods::as(x, "dgCMatrix")
y <- 1 + as.vector(x %*% c(1.1, -.4, rep(0,p-2))) + rnorm(n, sd = .4)
t_sparse <- system.time({
  fit_sparse <- mrvcr::lmnet_sparse(x, y, pet = "ElasticNet", lambda = .03,
                                    alpha = .4, tol = 1e-8)
})
# Explicitly requested densification only for the benchmark reference.
xx <- as.matrix(x)
d <- as.data.frame(xx)
d$y <- y
form <- stats::reformulate(colnames(xx), response = "y")
t_dense <- system.time({
  fit_dense <- mrvcr::lmnet(form, d, pet = "ElasticNet", lambda = .03,
                             alpha = .4, method = "coordinate_descent", tol = 1e-8)
})
cat(sprintf("Sparse: %.3f seconds; dense: %.3f seconds\n",
            t_sparse[["elapsed"]],t_dense[["elapsed"]]))
cat(sprintf("Maximum coefficient discrepancy: %.5g\n",
            max(abs(unname(coef(fit_sparse)) - unname(coef(fit_dense))))))
cat(sprintf("CSC input size: %.3f MB; dense input size: %.3f MB\n",
            as.numeric(object.size(x))/2^20, as.numeric(object.size(xx))/2^20))
stopifnot(fit_sparse$converged, fit_dense$converged)
