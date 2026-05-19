#' Linear Regression Model
#'
#' This intentionally shadows stats::lm when mrvcr is attached.
#'
#' `compat = "rust"` is the Rust-first mode. It does not inherit from `"lm"` or
#' `"mlm"` and uses mrvcr S3 methods.
#'
#' `compat = "stats"` builds base-R QR/effects components and inherits from
#' `"lm"`/`"mlm"` so stats S3 methods can be used as fallbacks.
#'
#' @export
lm <- function(formula, data, subset, weights, na.action,
               method = "qr", model = TRUE, x = FALSE, y = FALSE,
               qr = TRUE, singular.ok = TRUE, contrasts = NULL,
               offset, ...) {
  ret.x <- x
  ret.y <- y
  cl <- match.call()
  dots <- list(...)

  tol <- if (!is.null(dots$tol)) as.numeric(dots$tol) else 1e-7
  pet <- if (!is.null(dots$pet)) dots$pet else "OLS"

  compat <- if (!is.null(dots$compat)) {
    match.arg(as.character(dots$compat), c("rust", "stats"))
  } else {
    "rust"
  }

  calc_se <- if (!is.null(dots$se)) isTRUE(dots$se) else TRUE

  mf <- match.call(expand.dots = FALSE)
  m <- match(
    c("formula", "data", "subset", "weights", "na.action", "offset"),
    names(mf),
    0L
  )
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())

  if (identical(method, "model.frame")) {
    return(mf)
  }

  mt <- attr(mf, "terms")
  y_obj <- stats::model.response(mf, "numeric")
  w <- as.vector(stats::model.weights(mf))

  if (!is.null(w) && !is.numeric(w)) {
    stop("'weights' must be a numeric vector", call. = FALSE)
  }

  if (!is.null(w) && any(w < 0 | is.na(w))) {
    stop("missing or negative weights not allowed", call. = FALSE)
  }

  offset_vec <- stats::model.offset(mf)

  mlm <- is.matrix(y_obj)
  ny <- if (mlm) nrow(y_obj) else length(y_obj)
  n_resp <- if (mlm) ncol(y_obj) else 1L

  if (!is.null(offset_vec)) {
    offset_vec <- as.vector(offset_vec)

    if (NROW(offset_vec) != ny) {
      stop(
        sprintf(
          "number of offsets is %d, should equal %d (number of observations)",
          NROW(offset_vec),
          ny
        ),
        call. = FALSE
      )
    }
  }

  if (is.empty.model(mt)) {
    x_mat <- matrix(, ny, 0L)

    if (mlm) {
      resp_names <- colnames(y_obj) %||% paste0("Y", seq_len(n_resp))

      coef <- matrix(
        numeric(),
        nrow = 0L,
        ncol = n_resp,
        dimnames = list(NULL, resp_names)
      )

      fitted <- matrix(
        0,
        nrow = ny,
        ncol = n_resp,
        dimnames = list(rownames(mf), resp_names)
      )

      residuals <- y_obj
      rownames(residuals) <- rownames(mf)
      colnames(residuals) <- resp_names
    } else {
      coef <- numeric()
      fitted <- 0 * y_obj
      residuals <- y_obj

      names(fitted) <- rownames(mf)
      names(residuals) <- rownames(mf)
    }

    if (!is.null(offset_vec)) {
      if (mlm) {
        fitted <- matrix(
          offset_vec,
          nrow = ny,
          ncol = n_resp,
          dimnames = dimnames(fitted)
        )
        residuals <- y_obj - fitted
      } else {
        fitted <- offset_vec
        residuals <- y_obj - offset_vec
        names(fitted) <- rownames(mf)
        names(residuals) <- rownames(mf)
      }
    }

    z <- list(
      coefficients = coef,
      residuals = residuals,
      effects = NULL,
      rank = 0L,
      fitted.values = fitted,
      assign = attr(x_mat, "assign"),
      qr = NULL,
      df.residual = if (!is.null(w)) sum(w != 0) else ny,
      rust = NULL
    )
  } else {
    x_mat <- stats::model.matrix(mt, mf, contrasts)

    calc_se_rust <- isTRUE(calc_se) && !mlm

    z_rust <- if (is.null(w)) {
      lm_fit_rust(
        y_obj,
        x_mat,
        offset_vec,
        as.character(pet),
        as.character(method),
        as.numeric(tol),
        as.logical(singular.ok),
        as.logical(calc_se_rust)
      )
    } else {
      wlm_fit_rust(
        y_obj,
        x_mat,
        w,
        offset_vec,
        as.character(if (identical(pet, "OLS")) "WLS" else pet),
        as.character(method),
        as.numeric(tol),
        as.logical(singular.ok),
        as.logical(calc_se_rust)
      )
    }

    p <- ncol(x_mat)
    coef_names <- colnames(x_mat) %||% paste0("x", seq_len(p))
    resp_names <- if (mlm) {
      colnames(y_obj) %||% paste0("Y", seq_len(n_resp))
    } else {
      NULL
    }

    if (mlm) {
      coef <- matrix(
        z_rust$coefficients,
        nrow = p,
        ncol = n_resp,
        dimnames = list(coef_names, resp_names)
      )

      residuals <- matrix(
        z_rust$residuals,
        nrow = ny,
        ncol = n_resp,
        dimnames = list(rownames(mf), resp_names)
      )

      fitted <- matrix(
        z_rust$fitted_values,
        nrow = ny,
        ncol = n_resp,
        dimnames = list(rownames(mf), resp_names)
      )

      rust_effects <- if (isTRUE(z_rust$has_effects)) {
        matrix(
          z_rust$effects,
          nrow = ny,
          ncol = n_resp,
          dimnames = list(rownames(mf), resp_names)
        )
      } else {
        NULL
      }
    } else {
      coef <- z_rust$coefficients
      names(coef) <- coef_names

      residuals <- z_rust$residuals
      fitted <- z_rust$fitted_values

      names(residuals) <- rownames(mf)
      names(fitted) <- rownames(mf)

      rust_effects <- if (isTRUE(z_rust$has_effects)) {
        z_rust$effects
      } else {
        NULL
      }

      if (!is.null(rust_effects)) {
        names(rust_effects) <- rownames(mf)
      }
    }

    z_rust$effects_shaped <- rust_effects

    rank <- as.integer(z_rust$rank)

    if (!singular.ok && rank < ncol(x_mat)) {
      stop("singular fit encountered", call. = FALSE)
    }

    df.residual <- if (!is.null(w)) {
      sum(w != 0) - rank
    } else {
      ny - rank
    }

    if (identical(compat, "stats") && isTRUE(qr)) {
      w_eff <- if (is.null(w)) rep(1, ny) else w
      offset_eff <- if (is.null(offset_vec)) rep(0, ny) else offset_vec

      x_qr <- x_mat * sqrt(w_eff)

      if (mlm) {
        y_qr <- sweep(y_obj, 1L, offset_eff, "-")
        y_qr <- y_qr * sqrt(w_eff)
      } else {
        y_qr <- (y_obj - offset_eff) * sqrt(w_eff)
      }

      qr_obj <- base::qr(x_qr)
      effects <- base::qr.qty(qr_obj, y_qr)
      rank <- qr_obj$rank
      qr_component <- structure(qr_obj, class = "qr")
    } else {
      effects <- rust_effects
      qr_component <- NULL
    }

    z <- list(
      coefficients = coef,
      residuals = residuals,
      effects = effects,
      rank = rank,
      fitted.values = fitted,
      assign = attr(x_mat, "assign"),
      qr = qr_component,
      df.residual = df.residual,
      rust = z_rust
    )
  }

  if (mlm) {
    class(z) <- if (identical(compat, "stats")) {
      c("mrvcr_mlm", "mlm", "mrvcr_lm", "lm")
    } else {
      c("mrvcr_mlm", "mrvcr_lm")
    }
  } else {
    class(z) <- if (identical(compat, "stats")) {
      c("mrvcr_lm", "lm")
    } else {
      "mrvcr_lm"
    }
  }

  z$na.action <- attr(mf, "na.action")
  z$offset <- offset_vec
  z$contrasts <- attr(x_mat, "contrasts")
  z$xlevels <- stats::.getXlevels(mt, mf)
  z$call <- cl
  z$terms <- mt
  z$method <- method
  z$compat <- compat

  if (model) {
    z$model <- mf
  }

  if (ret.x) {
    z$x <- x_mat
  }

  if (ret.y) {
    z$y <- y_obj
  }

  if (!qr) {
    z$qr <- NULL
  }

  z
}
