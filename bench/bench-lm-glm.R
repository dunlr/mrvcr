# bench/bench-lm-glm.R
#
# Recommended install workflow on Windows:
# 1. Close R/RStudio/VS Code sessions if mrvcr was loaded.
# 2. Reopen R.
# 3. devtools::uninstall()
# 4. Restart R
# 5. devtools::clean_dll()
# 6. devtools::document()
# 7. Restart R
# 8. devtools::document()
# 9. devtools::install(upgrade = FALSE, build_vignettes = FALSE, args = c("--preclean"))
# 10. Restart R.
# 11. library(mrvcr)
#
# Do not benchmark from devtools::load_all().

library(mrvcr)
library(bench)

make_lm_data <- function(n = 10000, p = 20, seed = 1) {
  set.seed(seed)

  X_raw <- matrix(rnorm(n * p), nrow = n, ncol = p)
  colnames(X_raw) <- paste0("x", seq_len(p))

  beta <- seq_len(p) / p
  y <- drop(X_raw %*% beta + rnorm(n))

  df <- as.data.frame(X_raw)
  df$y <- y
  df$w <- seq_len(n)

  X <- cbind("(Intercept)" = 1, X_raw)

  list(
    df = df,
    y = y,
    X = X,
    w = df$w
  )
}

make_ill_conditioned_lm_data <- function(n = 10000, p = 20, eps = 1e-10, seed = 1) {
  set.seed(seed)

  x1 <- rnorm(n)
  X_raw <- matrix(rnorm(n * p), nrow = n, ncol = p)

  X_raw[, 1] <- x1
  X_raw[, 2] <- x1 + eps * rnorm(n)

  colnames(X_raw) <- paste0("x", seq_len(p))

  beta <- seq_len(p) / p
  y <- drop(X_raw %*% beta + rnorm(n, sd = 0.1))

  list(
    y = y,
    X = cbind("(Intercept)" = 1, X_raw)
  )
}

max_coef_diff <- function(a, b) {
  ok <- is.finite(a) & is.finite(b)

  if (!any(ok)) {
    return(NA_real_)
  }

  max(abs(a[ok] - b[ok]))
}

check_lm_correctness <- function(dat, tol = 1e-8) {
  fit_stats_lm <- stats::lm(y ~ . - w, data = dat$df)
  fit_stats_fit <- stats::lm.fit(x = dat$X, y = dat$y)

  fit_mrvcr_lm <- mrvcr::lm(y ~ . - w, data = dat$df)

  fit_rust_qr <- mrvcr:::lm_fit_rust(
    dat$y, dat$X, NULL, "OLS", "qr", 1e-7, TRUE, FALSE
  )

  fit_rust_chol <- mrvcr:::lm_fit_rust(
    dat$y, dat$X, NULL, "OLS", "cholesky", 1e-7, TRUE, FALSE
  )

  fit_rust_cpqr <- mrvcr:::lm_fit_rust(
    dat$y, dat$X, NULL, "OLS", "col_piv_qr", 1e-7, TRUE, FALSE
  )

  stopifnot(max(abs(coef(fit_stats_lm) - coef(fit_mrvcr_lm))) < tol)
  stopifnot(max(abs(fit_stats_fit$coefficients - fit_rust_qr$coefficients)) < tol)
  stopifnot(max(abs(fit_rust_qr$coefficients - fit_rust_chol$coefficients)) < 1e-7)
  stopifnot(max(abs(fit_rust_qr$coefficients - fit_rust_cpqr$coefficients)) < 1e-7)

  fit_stats_wfit <- stats::lm.wfit(x = dat$X, y = dat$y, w = dat$w)

  fit_rust_w_qr <- mrvcr:::wlm_fit_rust(
    dat$y, dat$X, dat$w, NULL, "WLS", "qr", 1e-7, TRUE, FALSE
  )

  fit_rust_w_chol <- mrvcr:::wlm_fit_rust(
    dat$y, dat$X, dat$w, NULL, "WLS", "cholesky", 1e-7, TRUE, FALSE
  )

  fit_rust_w_cpqr <- mrvcr:::wlm_fit_rust(
    dat$y, dat$X, dat$w, NULL, "WLS", "col_piv_qr", 1e-7, TRUE, FALSE
  )

  stopifnot(max(abs(fit_stats_wfit$coefficients - fit_rust_w_qr$coefficients)) < tol)
  stopifnot(max(abs(fit_rust_w_qr$coefficients - fit_rust_w_chol$coefficients)) < 1e-7)
  stopifnot(max(abs(fit_rust_w_qr$coefficients - fit_rust_w_cpqr$coefficients)) < 1e-7)

  invisible(TRUE)
}

# ------------------------------------------------------------------------------
# 1. Main benchmark
# ------------------------------------------------------------------------------

dat_main <- make_lm_data(n = 10000, p = 20, seed = 1)
check_lm_correctness(dat_main)

main_results <- bench::mark(
  stats_lm_formula = {
    stats::lm(y ~ . - w, data = dat_main$df)
  },

  stats_lm_fit = {
    stats::lm.fit(x = dat_main$X, y = dat_main$y)
  },

  mrvcr_lm_formula = {
    mrvcr::lm(y ~ . - w, data = dat_main$df)
  },

  mrvcr_lm_no_qr = {
    mrvcr::lm(
      y ~ . - w,
      data = dat_main$df,
      qr = FALSE,
      model = FALSE,
      x = FALSE,
      y = FALSE
    )
  },

  direct_rust_lm_qr = {
    mrvcr:::lm_fit_rust(
      dat_main$y, dat_main$X, NULL, "OLS", "qr", 1e-7, TRUE, FALSE
    )
  },

  direct_rust_lm_col_piv_qr = {
    mrvcr:::lm_fit_rust(
      dat_main$y, dat_main$X, NULL, "OLS", "col_piv_qr", 1e-7, TRUE, FALSE
    )
  },

  direct_rust_lm_cholesky = {
    mrvcr:::lm_fit_rust(
      dat_main$y, dat_main$X, NULL, "OLS", "cholesky", 1e-7, TRUE, FALSE
    )
  },

  stats_wlm_formula = {
    stats::lm(y ~ . - w, data = dat_main$df, weights = w)
  },

  stats_lm_wfit = {
    stats::lm.wfit(x = dat_main$X, y = dat_main$y, w = dat_main$w)
  },

  mrvcr_wlm_formula = {
    mrvcr::lm(y ~ . - w, data = dat_main$df, weights = w)
  },

  direct_rust_wlm_qr = {
    mrvcr:::wlm_fit_rust(
      dat_main$y, dat_main$X, dat_main$w, NULL, "WLS", "qr", 1e-7, TRUE, FALSE
    )
  },

  direct_rust_wlm_col_piv_qr = {
    mrvcr:::wlm_fit_rust(
      dat_main$y, dat_main$X, dat_main$w, NULL, "WLS", "col_piv_qr", 1e-7, TRUE, FALSE
    )
  },

  direct_rust_wlm_cholesky = {
    mrvcr:::wlm_fit_rust(
      dat_main$y, dat_main$X, dat_main$w, NULL, "WLS", "cholesky", 1e-7, TRUE, FALSE
    )
  },

  iterations = 20,
  check = FALSE
)

cat("\nMain summary:\n")
main_summary <- main_results[, c("expression", "median", "mem_alloc", "itr/sec")]
print(main_summary)

# ------------------------------------------------------------------------------
# 2. Grid benchmark
# ------------------------------------------------------------------------------

sizes <- expand.grid(
  n = c(1000, 10000, 50000, 100000),
  p = c(5, 20, 100),
  eps = c(1e-7)
)

all_results <- vector("list", nrow(sizes))

for (i in seq_len(nrow(sizes))) {
  n_i <- sizes$n[i]
  p_i <- sizes$p[i]
  eps_i <- sizes$eps[i]

  message("Benchmarking n = ", n_i, ", p = ", p_i, ", eps = ", eps_i)

  dat_grid <- make_lm_data(n = n_i, p = p_i, seed = 123)

  all_results[[i]] <- bench::mark(
    stats_lm_fit = {
      stats::lm.fit(x = dat_grid$X, y = dat_grid$y, tol = eps_i)
    },

    direct_rust_lm_qr = {
      mrvcr:::lm_fit_rust(
        dat_grid$y, dat_grid$X, NULL, "OLS", "qr", eps_i, TRUE, FALSE
      )
    },

    direct_rust_lm_col_piv_qr = {
      mrvcr:::lm_fit_rust(
        dat_grid$y, dat_grid$X, NULL, "OLS", "col_piv_qr", eps_i, TRUE, FALSE
      )
    },

    direct_rust_lm_cholesky = {
      mrvcr:::lm_fit_rust(
        dat_grid$y, dat_grid$X, NULL, "OLS", "cholesky", eps_i, TRUE, FALSE
      )
    },

    stats_lm_wfit = {
      stats::lm.wfit(x = dat_grid$X, y = dat_grid$y, w = dat_grid$w, tol = eps_i)
    },

    direct_rust_wlm_qr = {
      mrvcr:::wlm_fit_rust(
        dat_grid$y, dat_grid$X, dat_grid$w, NULL, "WLS", "qr", eps_i, TRUE, FALSE
      )
    },

    direct_rust_wlm_col_piv_qr = {
      mrvcr:::wlm_fit_rust(
        dat_grid$y, dat_grid$X, dat_grid$w, NULL, "WLS", "col_piv_qr", eps_i, TRUE, FALSE
      )
    },

    direct_rust_wlm_cholesky = {
      mrvcr:::wlm_fit_rust(
        dat_grid$y, dat_grid$X, dat_grid$w, NULL, "WLS", "cholesky", eps_i, TRUE, FALSE
      )
    },

    iterations = 5,
    check = FALSE
  )

  all_results[[i]]$n <- n_i
  all_results[[i]]$p <- p_i
  all_results[[i]]$eps <- eps_i
}

grid_results <- do.call(rbind, all_results)
grid_summary <- grid_results[, c("expression", "n", "p", "eps", "min", "median", "mem_alloc", "itr/sec")]

cat("\nGrid summary:\n")
print(grid_summary, n = Inf)

# ------------------------------------------------------------------------------
# 3. Zero-weight benchmark
# ------------------------------------------------------------------------------

dat_zero <- make_lm_data(n = 50000, p = 20, seed = 456)
dat_zero$w[seq(1, length(dat_zero$w), by = 5)] <- 0

zero_results <- bench::mark(
  stats_lm_wfit_zero = {
    stats::lm.wfit(x = dat_zero$X, y = dat_zero$y, w = dat_zero$w)
  },

  direct_rust_wlm_qr_zero = {
    mrvcr:::wlm_fit_rust(
      dat_zero$y, dat_zero$X, dat_zero$w, NULL, "WLS", "qr", 1e-7, TRUE, FALSE
    )
  },

  direct_rust_wlm_col_piv_qr_zero = {
    mrvcr:::wlm_fit_rust(
      dat_zero$y, dat_zero$X, dat_zero$w, NULL, "WLS", "col_piv_qr", 1e-7, TRUE, FALSE
    )
  },

  direct_rust_wlm_cholesky_zero = {
    mrvcr:::wlm_fit_rust(
      dat_zero$y, dat_zero$X, dat_zero$w, NULL, "WLS", "cholesky", 1e-7, TRUE, FALSE
    )
  },

  iterations = 10,
  check = FALSE
)

cat("\nZero-weight summary:\n")
zero_summary <- zero_results[, c("expression", "median", "mem_alloc", "itr/sec")]
print(zero_summary)

# ------------------------------------------------------------------------------
# 4. Ill-conditioned diagnostic
# ------------------------------------------------------------------------------

cat("\nIll-conditioned design diagnostic:\n")

for (eps in c(1e-4, 1e-6, 1e-8, 1e-10)) {
  dat_bad <- make_ill_conditioned_lm_data(eps = eps)

  fit_qr <- tryCatch(
    mrvcr:::lm_fit_rust(dat_bad$y, dat_bad$X, NULL, "OLS", "qr", eps, TRUE, FALSE),
    error = identity
  )

  fit_cpqr <- tryCatch(
    mrvcr:::lm_fit_rust(dat_bad$y, dat_bad$X, NULL, "OLS", "col_piv_qr", eps, TRUE, FALSE),
    error = identity
  )

  fit_chol <- tryCatch(
    mrvcr:::lm_fit_rust(dat_bad$y, dat_bad$X, NULL, "OLS", "cholesky", eps, TRUE, FALSE),
    error = identity
  )

  fit_stats <- stats::lm.fit(dat_bad$X, dat_bad$y, tol = eps)

  cat("\neps =", eps, "\n")
  cat("stats rank:", fit_stats$rank, "\n")
  cat("stats has NA coefficients:", anyNA(fit_stats$coefficients), "\n")

  if (inherits(fit_qr, "error")) {
    cat("qr error:", conditionMessage(fit_qr), "\n")
  } else {
    cat("qr rank:", fit_qr$rank, "\n")
    cat("qr has NA coefficients:", anyNA(fit_qr$coefficients), "\n")
    cat("qr diff vs stats:", max_coef_diff(fit_qr$coefficients, fit_stats$coefficients), "\n")
  }

  if (inherits(fit_cpqr, "error")) {
    cat("col_piv_qr error:", conditionMessage(fit_cpqr), "\n")
  } else {
    cat("col_piv_qr rank:", fit_cpqr$rank, "\n")
    cat("col_piv_qr has NA coefficients:", anyNA(fit_cpqr$coefficients), "\n")
    cat("col_piv_qr pivot:", paste(fit_cpqr$pivot, collapse = ","), "\n")
    cat("col_piv_qr NA pattern:", paste(is.na(fit_cpqr$coefficients), collapse = ","), "\n")
    cat("stats NA pattern:", paste(is.na(fit_stats$coefficients), collapse = ","), "\n")
    cat("col_piv_qr diff vs stats:", max_coef_diff(fit_cpqr$coefficients, fit_stats$coefficients), "\n")
    cat("col_piv_qr fitted diff vs stats:", max(abs(fit_cpqr$fitted_values - fit_stats$fitted.values)), "\n")
    cat("col_piv_qr residual diff vs stats:", max(abs(fit_cpqr$residuals - fit_stats$residuals)), "\n")
    cat("col_piv_qr RSS:", sum(fit_cpqr$residuals^2), "\n")
    cat("stats RSS:", sum(fit_stats$residuals^2), "\n")
  }

  if (inherits(fit_chol, "error")) {
    cat("cholesky error:", conditionMessage(fit_chol), "\n")
  } else {
    cat("cholesky rank:", fit_chol$rank, "\n")
    cat("cholesky has NA coefficients:", anyNA(fit_chol$coefficients), "\n")
    cat("cholesky diff vs stats:", max_coef_diff(fit_chol$coefficients, fit_stats$coefficients), "\n")
  }
}

# ------------------------------------------------------------------------------
# 5. Multivariate response benchmark
# ------------------------------------------------------------------------------

make_mlm_data <- function(n = 10000, p = 20, r = 5, seed = 1) {
  set.seed(seed)

  X_raw <- matrix(rnorm(n * p), nrow = n, ncol = p)
  colnames(X_raw) <- paste0("x", seq_len(p))

  X <- cbind("(Intercept)" = 1, X_raw)
  B <- matrix(rnorm((p + 1) * r), nrow = p + 1, ncol = r)
  Y <- X %*% B + matrix(rnorm(n * r), nrow = n)

  W <- seq_len(n)

  list(X = X, Y = Y, W = W)
}

mlm_dat <- make_mlm_data(n = 10000, p = 100, r = 5)

mlm_results <- bench::mark(
  stats_lm_fit_mlm = {
    stats::lm.fit(mlm_dat$X, mlm_dat$Y)
  },

  direct_rust_lm_qr_mlm = {
    mrvcr:::lm_fit_rust(
      mlm_dat$Y, mlm_dat$X, NULL, "OLS", "qr", 1e-7, TRUE, FALSE
    )
  },

  direct_rust_lm_col_piv_qr_mlm = {
    mrvcr:::lm_fit_rust(
      mlm_dat$Y, mlm_dat$X, NULL, "OLS", "col_piv_qr", 1e-7, TRUE, FALSE
    )
  },

  direct_rust_lm_cholesky_mlm = {
    mrvcr:::lm_fit_rust(
      mlm_dat$Y, mlm_dat$X, NULL, "OLS", "cholesky", 1e-7, TRUE, FALSE
    )
  },

  stats_lm_wfit_mlm = {
    stats::lm.wfit(mlm_dat$X, mlm_dat$Y, mlm_dat$W)
  },

  direct_rust_wlm_qr_mlm = {
    mrvcr:::wlm_fit_rust(
      mlm_dat$Y, mlm_dat$X, mlm_dat$W, NULL, "WLS", "qr", 1e-7, TRUE, FALSE
    )
  },

  direct_rust_wlm_col_piv_qr_mlm = {
    mrvcr:::wlm_fit_rust(
      mlm_dat$Y, mlm_dat$X, mlm_dat$W, NULL, "WLS", "col_piv_qr", 1e-7, TRUE, FALSE
    )
  },

  direct_rust_wlm_cholesky_mlm = {
    mrvcr:::wlm_fit_rust(
      mlm_dat$Y, mlm_dat$X, mlm_dat$W, NULL, "WLS", "cholesky", 1e-7, TRUE, FALSE
    )
  },

  iterations = 10,
  check = FALSE
)

cat("\nMultivariate response benchmark:\n")
print(mlm_results[, c("expression", "median", "mem_alloc", "itr/sec")])

invisible(list(
  main = main_results,
  grid = grid_results,
  zero_weights = zero_results,
  multivariate = mlm_results
))


bench::mark(
  direct_rust_wlm_qr_zero = {
    mrvcr:::wlm_fit_rust(
      dat_zero$y, dat_zero$X, dat_zero$w, NULL, "WLS", "householder_qr", 1e-7, TRUE, FALSE
    )
  },
  direct_rust_wlm_col_piv_qr_zero = {
    mrvcr:::wlm_fit_rust(
      dat_zero$y, dat_zero$X, dat_zero$w, NULL, "WLS", "col_piv_qr", 1e-7, TRUE, FALSE
    )
  },
  direct_rust_wlm_cholesky_zero = {
    mrvcr:::wlm_fit_rust(
      dat_zero$y, dat_zero$X, dat_zero$w, NULL, "WLS", "cholesky", 1e-7, TRUE, FALSE
    )
  },
  iterations = 100,
  check = FALSE
)
