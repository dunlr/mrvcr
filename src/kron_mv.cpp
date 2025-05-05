#include <RcppArmadillo.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo)]]

// Efficient product: (sum_i Σ[i] ⊗ V[i]) %*% x without forming full Ω
// [[Rcpp::export]]
arma::vec kron_mv(const List& sigmas, const List& Vs, const arma::vec& x) {
  int m = sigmas.size();
  arma::vec out = arma::zeros<arma::vec>(x.n_elem);
  for (int i = 0; i < m; ++i) {
    arma::mat Si = sigmas[i];
    arma::mat Vi = Vs[i];
    int n = Vi.n_rows;
    int d = Si.n_rows;
    // reinterpret the const data pointer as non-const for Armadillo
    double* ptr = const_cast<double*>( x.begin() );
    arma::mat Xmat(ptr, n, d, false, false);
    arma::mat tmp = Vi * Xmat * Si.t();
    out += arma::vectorise(tmp);
  }
  return out;
}
