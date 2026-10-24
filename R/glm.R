#' Generalized Linear Model
#'
#' This intentionally shadows stats::glm when mrvcr is attached.
#' The returned object inherits from c("glm", "lm"), so base R S3 methods are used.
#'
#' @param formula A model formula, as in [stats::glm()].
#' @param family An R family object, family constructor, or family name. Native
#'   family and link support is validated by the Rust backend.
#' @param data An optional data frame, list, or environment containing variables.
#' @param weights Optional prior weights.
#' @param subset An optional expression selecting observations.
#' @param na.action A missing-value handling function.
#' @param start Optional starting coefficients; currently delegated to [stats::glm()].
#' @param etastart Optional initial linear predictor; currently delegated to [stats::glm()].
#' @param mustart Optional initial fitted mean; currently delegated to [stats::glm()].
#' @param offset Optional offset added to the linear predictor.
#' @param control A list of standard GLM control parameters.
#' @param model Whether to retain the model frame in the result.
#' @param method Standard fitting entry-point name; only `"glm.fit"` uses Rust.
#' @param x Whether to retain the model matrix.
#' @param y Whether to retain the response.
#' @param singular.ok Whether a rank-deficient design is permitted.
#' @param contrasts Optional factor contrasts for the model matrix.
#' @param ... Additional backend controls (e.g. `pet`, `rust_method`, `param`)
#'   and GLM control arguments.
#'
#' @importFrom stats gaussian
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
  # A family object is the source of truth. Parameterized families expose
  # their parameters inside the variance function closure (MASS / statmod).
  # No duplicate list of Rust-supported families or links is maintained in R.
  param <- if (!is.null(dots$param)) {
    as.numeric(dots$param)
  } else {
    NULL
  }

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

  if (!inherits(family, "family") || length(family$family) != 1L ||
      length(family$link) != 1L) {
    stop("'family' must be a valid R family object", call. = FALSE)
  }

  # Extract only the scalar distribution parameter required by the Rust ABI.
  # Unsupported family/link names are rejected by Rust enum pattern matching.
  family_name <- as.character(family$family)
  if (is.null(param)) {
    venv <- environment(family$variance)
    if (startsWith(tolower(family_name), "negative binomial(") &&
        is.environment(venv) &&
        exists(".Theta", envir = venv, inherits = FALSE)) {
      param <- get(".Theta", envir = venv, inherits = FALSE)
    } else if (identical(tolower(family_name), "tweedie") &&
               is.environment(venv) &&
               exists("p", envir = venv, inherits = FALSE)) {
      param <- get("p", envir = venv, inherits = FALSE)
    } else {
      param <- 0
    }
  }
  if (!is.numeric(param) || length(param) != 1L || !is.finite(param)) {
    stop("Family parameter 'param' must be a finite numeric scalar", call. = FALSE)
  }

  # Guard against accidentally passing one variance parameter to R and
  # a different one to Rust (important for correct deviance and inference).
  if (startsWith(tolower(family_name), "negative binomial(") &&
      is.finite(param)) {
    actual_var <- family$variance(1)
    expected_var <- 1 + 1 / param
    if (!isTRUE(all.equal(actual_var, expected_var, tolerance = 1e-10))) {
      stop("Negative binomial 'param' differs from family theta", call. = FALSE)
    }
  } else if (identical(tolower(family_name), "tweedie")) {
    if (!isTRUE(all.equal(family$variance(2), 2^param, tolerance = 1e-10))) {
      stop("Tweedie 'param' differs from family var.power", call. = FALSE)
    }
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

  # Extract the response once. Two-column binomial outcomes are converted to
  # proportions with trial counts as prior weights (stats::glm convention).
  trials <- NULL
  if (is.matrix(Y) && NCOL(Y) == 2L && identical(family_name, "binomial")) {
    if (!is.numeric(Y) || any(!is.finite(Y)) || any(Y < 0) ||
        any(Y != floor(Y))) {
      stop("Two-column binomial responses need nonnegative integer counts", call. = FALSE)
    }
    trials <- rowSums(Y)
    Y <- ifelse(trials > 0, Y[, 1L] / pmax(trials, 1), 0)
  } else if (is.matrix(Y) && NCOL(Y) > 1L) {
    stop("Matrix-valued GLM responses are not implemented for this family", call. = FALSE)
  }

  if (is.factor(Y)) {
    if (!identical(family_name, "binomial") || nlevels(Y) != 2L) {
      stop("Factor-valued responses are supported only for binary binomial GLMs", call. = FALSE)
    }
    Y <- as.numeric(Y) - 1
  }

  if (length(dim(Y)) == 1L) {
    nm <- names(Y)
    dim(Y) <- NULL
    if (!is.null(nm)) names(Y) <- nm
  }

  X <- if (!stats::is.empty.model(mt)) {
    stats::model.matrix(mt, mf, contrasts)
  } else {
    matrix(, NROW(Y), 0L)
  }

  w <- as.vector(stats::model.weights(mf))

  if (!is.null(w) && !is.numeric(w)) {
    stop("'weights' must be a numeric vector", call. = FALSE)
  }

  if (!is.null(w) && any(!is.finite(w))) {
    stop("weights must be finite", call. = FALSE)
  }
  if (!is.null(w) && any(w < 0)) {
    stop("negative weights not allowed", call. = FALSE)
  }
  if (!is.null(trials)) {
    w <- if (is.null(w)) trials else w * trials
  }
  if (identical(family_name, "binomial") &&
      (any(!is.finite(Y)) || any(Y < 0 | Y > 1))) {
    stop("binomial responses must be proportions in [0, 1]", call. = FALSE)
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
    family_name,
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
