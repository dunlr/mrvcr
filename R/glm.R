#' Generalized Linear Model
#'
#' This intentionally shadows stats::glm when mrvcr is attached.
#' The returned object inherits from c("glm", "lm"), so base R S3 methods are used.
#'
#' @export
glm <- function(formula, family = gaussian, data, weights, subset,
                na.action, start = NULL, etastart, mustart, offset,
                control = list(...), model = TRUE, method = "glm.fit",
                x = FALSE, y = TRUE, singular.ok = TRUE,
                contrasts = NULL, ...) {
  cal <- match.call()
  dots <- list(...)

  pet <- if (!is.null(dots$pet)) dots$pet else "MLE"
  rust_method <- if (!is.null(dots$rust_method)) dots$rust_method else "FisherScoring"
  param <- if (!is.null(dots$param)) dots$param else 0

  backend_arg_names <- c("pet", "rust_method", "param")

  if (missing(control)) {
    control <- dots[setdiff(names(dots), backend_arg_names)]
  }

  if (is.character(family)) {
    family <- get(family, mode = "function", envir = parent.frame())
  }

  if (is.function(family)) {
    family <- family()
  }

  if (is.null(family$family)) {
    print(family)
    stop("'family' not recognized", call. = FALSE)
  }

  if (!identical(method, "glm.fit")) {
    if (!identical(method, "model.frame")) {
      cal[[1L]] <- quote(stats::glm)
      warning(
        "mrvcr::glm currently uses the Rust backend only for method = 'glm.fit'; using stats::glm().",
        call. = FALSE
      )
      return(eval(cal, parent.frame()))
    }
  }

  if (!is.null(start) || !missing(etastart) || !missing(mustart)) {
    cal[[1L]] <- quote(stats::glm)
    warning(
      "mrvcr::glm does not yet support start/etastart/mustart; using stats::glm().",
      call. = FALSE
    )
    return(eval(cal, parent.frame()))
  }

  control <- do.call(stats::glm.control, control)

  mf <- match.call(expand.dots = FALSE)
  m <- match(
    c("formula", "data", "subset", "weights", "na.action",
      "etastart", "mustart", "offset"),
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
  Y <- stats::model.response(mf, "any")

  if (is.matrix(Y) && NCOL(Y) > 1L) {
    cal[[1L]] <- quote(stats::glm)
    warning(
      "mrvcr::glm does not yet support matrix/binomial two-column responses; using stats::glm().",
      call. = FALSE
    )
    return(eval(cal, parent.frame()))
  }

  if (length(dim(Y)) == 1L) {
    nm <- rownames(Y)
    dim(Y) <- NULL

    if (!is.null(nm)) {
      names(Y) <- nm
    }
  }

  X <- if (!is.empty.model(mt)) {
    stats::model.matrix(mt, mf, contrasts)
  } else {
    matrix(, NROW(Y), 0L)
  }

  w <- as.vector(stats::model.weights(mf))

  if (!is.null(w) && !is.numeric(w)) {
    stop("'weights' must be a numeric vector", call. = FALSE)
  }

  if (!is.null(w) && any(w < 0)) {
    stop("negative weights not allowed", call. = FALSE)
  }

  offset_vec <- as.vector(stats::model.offset(mf))

  if (!is.null(offset_vec)) {
    if (length(offset_vec) != NROW(Y)) {
      stop(
        sprintf(
          "number of offsets is %d should equal %d (number of observations)",
          length(offset_vec),
          NROW(Y)
        ),
        call. = FALSE
      )
    }
  }

  fit_rust <- glm_fit_rust(
    Y,
    X,
    w,
    offset_vec,
    as.character(family$family),
    as.character(family$link),
    as.character(pet),
    as.character(rust_method),
    as.numeric(param),
    as.integer(control$maxit),
    as.numeric(control$epsilon),
    as.logical(singular.ok),
    FALSE
  )

  coefficients <- fit_rust$coefficients
  names(coefficients) <- colnames(X) %||% paste0("x", seq_along(coefficients))

  fitted_values <- fit_rust$fitted_values
  linear_predictors <- fit_rust$linear_predictors
  residual_values <- fit_rust$residuals

  names(fitted_values) <- rownames(mf)
  names(linear_predictors) <- rownames(mf)
  names(residual_values) <- rownames(mf)

  prior_weights <- if (is.null(w)) rep(1, NROW(Y)) else w
  working_weights <- if (!is.null(fit_rust$weights)) fit_rust$weights else prior_weights

  x_qr <- X * sqrt(working_weights)
  qr_obj <- base::qr(x_qr)
  rank <- qr_obj$rank

  if (!singular.ok && rank < ncol(X)) {
    stop("singular fit encountered", call. = FALSE)
  }

  has_intercept <- attr(mt, "intercept") > 0L
  n_eff <- sum(prior_weights != 0)

  deviance <- sum(family$dev.resids(Y, fitted_values, prior_weights))

  null_deviance <- .mrvcr_glm_null_deviance(
    y = Y,
    family = family,
    weights = prior_weights,
    offset = offset_vec,
    intercept = has_intercept
  )

  aic <- .mrvcr_glm_aic(
    y = Y,
    family = family,
    fitted.values = fitted_values,
    weights = prior_weights,
    deviance = deviance,
    rank = rank
  )

  fit <- list(
    coefficients = coefficients,
    residuals = residual_values,
    fitted.values = fitted_values,
    effects = NULL,
    R = base::qr.R(qr_obj),
    rank = rank,
    qr = structure(qr_obj, class = "qr"),
    family = family,
    linear.predictors = linear_predictors,
    deviance = deviance,
    aic = aic,
    null.deviance = null_deviance,
    iter = fit_rust$iterations,
    weights = working_weights,
    prior.weights = prior_weights,
    df.residual = n_eff - rank,
    df.null = n_eff - as.integer(has_intercept),
    y = if (y) Y else NULL,
    converged = isTRUE(fit_rust$converged),
    boundary = FALSE,
    model = if (model) mf else NULL,
    call = cal,
    formula = formula,
    terms = mt,
    data = data,
    offset = offset_vec,
    control = control,
    method = method,
    contrasts = attr(X, "contrasts"),
    xlevels = stats::.getXlevels(mt, mf),
    rust = fit_rust
  )

  fit$na.action <- attr(mf, "na.action")

  if (x) {
    fit$x <- X
  }

  class(fit) <- c("mrvcr_glm", "glm", "lm")
  fit
}
