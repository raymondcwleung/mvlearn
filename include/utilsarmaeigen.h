#ifndef UTILS_ARMA_EIGEN_H
#define UTILS_ARMA_EIGEN_H

#include <Eigen/Dense>
#include <armadillo>

Eigen::MatrixXd cast_arma_to_eigen(const arma::Mat<double> &arma_mat);

arma::Mat<double> cast_eigen_to_arma(const Eigen::MatrixXd &eigen_mat);

#endif
