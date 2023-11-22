#ifndef UTILS_EIGENARMA_CONVERSIONS_H
#define UTILS_EIGENARMA_CONVERSIONS_H

#include <Eigen/Dense>
#include <armadillo>
#include <fstream>
#include <iostream>
#include <vector>

#include "Eigen/src/Core/Matrix.h"

namespace utilseigenarma {

//! Cast Armadillo double-valued matrices to Eigen double-valued matrices
/*!
 * \param arma_mat An Armadillo double matrix (note pass-by-value)
 *
 * \return An Eigen double matrix
 */
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> castArmaToEigen(
    arma::Mat<T> arma_mat) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat =
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
          arma_mat.memptr(), arma_mat.n_rows, arma_mat.n_cols);

  return eigen_mat;
};

template <typename T>
Eigen::Vector<T, Eigen::Dynamic> castArmaToEigen(arma::Col<T> arma_vec) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat =
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
          arma_vec.memptr(), arma_vec.n_rows, arma_vec.n_cols);

  return eigen_mat;
};

template <typename T>
Eigen::Vector<T, Eigen::Dynamic> castArmaToEigen(arma::Row<T> arma_vec) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat =
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
          arma_vec.memptr(), arma_vec.n_rows, arma_vec.n_cols);

  return eigen_mat;
};

//! Cast Eigen double-valued matrices to Armadillo double-valued matrices
/*!
 * \param eigen_mat A Eigen double matrix (note pass-by-value)
 *
 * \return An Armadillo double matrix
 */
template <typename T>
arma::Mat<T> castEigenToArma(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat) {
  arma::Mat<T> arma_mat =
      arma::Mat<T>(eigen_mat.data(), eigen_mat.rows(), eigen_mat.cols(),
                   true,  // true here is important due to pass-by-value
                   false);

  return arma_mat;
};

}  // namespace utilseigenarma

#endif
