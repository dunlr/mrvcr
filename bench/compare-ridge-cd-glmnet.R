# Exploratory diagnostic: DOES changing the optimizer explain raw-y glmnet gaps?
# Run from the package root. Only glmnet is optional.
# The previous analytical diagnostic already established mrvcr solves its own
# original-scale criterion; here we isolate the solver choice.

if (!requireNamespace("glmnet", quietly = TRUE)) {
  stop("Install glmnet to run this optional comparison.", call. = FALSE)
}

set.seed(849)
n <- 900L
p <- 15L
X <- matrix(rnorm(n*p), nrow = n, ncol = p)
colnames(X) <- paste0("x", seq_len(p))
truth <- c(1.5, -0.8, rep(0, p-2L))
y <- 2 + drop(X %*% truth) + rnorm(n, sd = 0.7)
d <- as.data.frame(X)
d$y <- y
predictor_formula <- stats::reformulate(colnames(X), response = "y")
lambda <- 0.12

for (alpha in c(0, 0.3)) {
  pet <- if (alpha == 0) "Ridge" else "ElasticNet"
  auto <- mrvcr::lmnet(predictor_formula, data = d,
                       pet = pet, lambda = lambda, alpha = alpha,
                       method = "auto", tol = 1e-10)
  cd <- mrvcr::lmnet(predictor_formula, data = d,
                     pet = pet, lambda = lambda, alpha = alpha,
                     method = "coordinate_descent", tol = 1e-10,
                     max_iter = 20000L)

  reference <- glmnet::glmnet(
    X, y, family = "gaussian", lambda = c(4, 2, 1)*lambda,
    alpha = alpha, intercept = TRUE, standardize = FALSE,
    control = list(thresh = 1e-12, maxit = 100000L)
  )
  ref <- as.vector(stats::coef(reference, s = lambda))
  b_auto <- unname(as.vector(stats::coef(auto)))
  b_cd <- unname(as.vector(stats::coef(cd)))
  stopifnot(length(ref) == p+1L, length(b_auto) == p+1L,
            length(b_cd) == p+1L, isTRUE(cd$converged))

  cat(sprintf("\n%-10s lambda=%.3f alpha=%.1f\n", pet, lambda, alpha))
  cat(sprintf("  auto method:                %s\n", auto$method))
  cat(sprintf("  explicit method:            %s\n", cd$method))
  cat(sprintf("  |beta(auto) - beta(CD)|:    %.12g\n", max(abs(b_auto-b_cd))))
  cat(sprintf("  |beta(CD) - beta(glmnet)|:  %.12g\n", max(abs(b_cd-ref))))
  cat(sprintf("  |beta(auto) - beta(glmnet)|:%.12g\n", max(abs(b_auto-ref))))
  cat(sprintf("  |objective(auto) - CD|:     %.12g\n", abs(auto$objective-cd$objective)))
  cat(sprintf("  CD KKT violation:           %.12g\n", cd$kkt_violation))
}

cat("\nIf the auto/CD gap is tiny but both have a larger glmnet gap,\n")
cat("the gap is not due to coordinate descent versus Cholesky.\n")
cat("Both implementations must be compared using the same effective objective.\n")
