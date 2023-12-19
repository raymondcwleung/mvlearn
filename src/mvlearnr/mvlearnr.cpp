
#include <Rcpp.h>

#include "Rcpp/vector/instantiation.h"
/* #include "cluster/mv_coreg_spectral.h" */
/* #include "cluster/mv_spectral.h" */

// [[Rcpp::export]]
Rcpp::NumericVector timesTwo(Rcpp::NumericVector x) { return x * 2; }
