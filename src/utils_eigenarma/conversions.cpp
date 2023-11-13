#include "conversions.h"

#include <Eigen/Dense>
#include <armadillo>
#include <fstream>
#include <iostream>
#include <vector>

namespace utilseigenarma {

//! Cast Armadillo double-valued matrices to Eigen double-valued matrices
/*!
 * \param arma_mat An Armadillo double matrix (note pass-by-value)
 *
 * \return An Eigen double matrix
 */
Eigen::MatrixXd castArmaToEigen(arma::Mat<double> arma_mat) {
  Eigen::MatrixXd eigen_mat = Eigen::Map<Eigen::MatrixXd>(
      arma_mat.memptr(), arma_mat.n_rows, arma_mat.n_cols);

  return eigen_mat;
};

//! Cast Eigen double-valued matrices to Armadillo double-valued matrices
/*!
 * \param eigen_mat A Eigen double matrix (note pass-by-value)
 *
 * \return An Armadillo double matrix
 */
arma::Mat<double> castEigenToArma(Eigen::MatrixXd eigen_mat) {
  arma::Mat<double> arma_mat = arma::Mat<double>(
      eigen_mat.data(), eigen_mat.rows(), eigen_mat.cols(), false, false);

  return arma_mat;
};

}  // namespace utilseigenarma
