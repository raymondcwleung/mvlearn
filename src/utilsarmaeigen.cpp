#include "utilsarmaeigen.h"

#include <Eigen/Dense>
#include <armadillo>

Eigen::MatrixXd cast_arma_to_eigen(arma::Mat<double>& arma_mat) {
  Eigen::MatrixXd eigen_mat = Eigen::Map<Eigen::MatrixXd>(
      arma_mat.memptr(), arma_mat.n_rows, arma_mat.n_cols);

  return eigen_mat;
};

arma::Mat<double> cast_eigen_to_arma(Eigen::MatrixXd& eigen_mat) {
  arma::Mat<double> arma_mat = arma::Mat<double>(
      eigen_mat.data(), eigen_mat.rows(), eigen_mat.cols(), false, false);

  return arma_mat;
};
