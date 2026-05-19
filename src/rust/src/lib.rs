// src/lib.rs
use extendr_api::prelude::*;

pub mod core;
pub mod models;

// this is a rust comment note that using `///` will generate documentation for the function, which can be viewed in R using ?hello_world

// Rust function that can be called from R. The `#[extendr]` attribute makes it available to R. The function returns a static string slice, which is a string literal in Rust. When called from R, it will return the string "Hello world!".

// --- LINEAR MODULE INTERFACES ---

#[extendr]
fn lm_fit_rust(
    y: Robj,
    x: Robj,
    offset: Robj,
    pet: &str,
    method: &str,
    tol: f64,
    singular_ok: bool,
    calc_se: bool,
) -> extendr_api::Result<List> {
    models::linear::lm::ffi_fit(
        y,
        x,
        offset,
        pet,
        method,
        tol,
        singular_ok,
        calc_se,
    )
}

#[extendr]
fn wlm_fit_rust(
    y: Robj,
    x: Robj,
    weights: Robj,
    offset: Robj,
    pet: &str,
    method: &str,
    tol: f64,
    singular_ok: bool,
    calc_se: bool,
) -> extendr_api::Result<List> {
    models::linear::lm::ffi_wfit(
        y,
        x,
        weights,
        offset,
        pet,
        method,
        tol,
        singular_ok,
        calc_se,
    )
}

#[extendr]
fn glm_fit_rust(
    y: Robj,
    x: Robj,
    weights: Robj,
    offset: Robj,
    family: &str,
    link: &str,
    pet: &str,
    method: &str,
    param: f64,
    max_iter: i32,
    tol: f64,
    singular_ok: bool,
    calc_se: bool,
) -> extendr_api::Result<List> {
    models::linear::glm::ffi_fit_standard(
        y,
        x,
        weights,
        offset,
        family,
        link,
        pet,
        method,
        param,
        max_iter,
        tol,
        singular_ok,
        calc_se,
    )
}

// --- VARIANCE COMPONENTS MODULE INTERFACES ---

#[extendr]
fn vcm_fit_rust(
    y: Robj,
    x: Robj,
    v: Robj,
    pet: &str,
    method: &str,
) -> extendr_api::Result<List> {
    models::vcomp::vcm::ffi_fit_general(y, x, v, pet, method)
}

#[extendr]
fn vcm_zzt_fit_rust(
    y: Robj,
    x: Robj,
    z: Robj,
    pet: &str,
    method: &str,
) -> extendr_api::Result<List> {
    models::vcomp::vcm::ffi_fit_zzt(y, x, z, pet, method)
}

#[extendr]
fn mvcm_fit_rust(
    y: Robj,
    x: Robj,
    pet: &str,
) -> extendr_api::Result<List> {
    models::vcomp::mvcm::ffi_fit(y, x, pet)
}

// --- PLACEHOLDERS ---

#[extendr]
fn lmm_fit_rust(
    _y: Robj,
    _x: Robj,
    _z: Robj,
    _pet: &str,
) -> extendr_api::Result<List> {
    Ok(list!(
        status = "LMM interface is registered, but the LMM solver is not implemented yet."
    ))
}

#[extendr]
fn glmm_fit_rust(
    _y: Robj,
    _x: Robj,
    _family: &str,
    _method: &str,
) -> extendr_api::Result<List> {
    Ok(list!(
        status = "GLMM interface is registered, but the GLMM solver is not implemented yet."
    ))
}

#[extendr]
fn arima_fit_rust(
    _y: Robj,
    _p: i32,
    _d: i32,
    _q: i32,
    _pet: &str,
) -> extendr_api::Result<List> {
    Ok(list!(
        status = "ARIMA interface is registered, but the ARIMA solver is not implemented yet."
    ))
}
// The `extendr_module!` macro is used to define a module that contains the exported functions. In this case, we define a module named `mrvcr` and include the `hello_world` function in it. This allows R to recognize and call the function when the package is loaded.

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod mrvcr;
    fn lm_fit_rust;
    fn wlm_fit_rust;
    fn glm_fit_rust;

    fn vcm_fit_rust;
    fn vcm_zzt_fit_rust;
    fn mvcm_fit_rust;

    fn lmm_fit_rust;
    fn glmm_fit_rust;
    fn arima_fit_rust;
}