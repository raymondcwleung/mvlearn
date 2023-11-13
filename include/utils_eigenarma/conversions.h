#ifndef UTILS_EIGENARMA_CONVERSIONS_H
#define UTILS_EIGENARMA_CONVERSIONS_H

#include <Eigen/Dense>
#include <armadillo>

namespace utilseigenarma {

Eigen::MatrixXd castArmaToEigen(arma::Mat<double> arma_mat);

arma::Mat<double> castEigenToArma(Eigen::MatrixXd eigen_mat);

}  // namespace utilseigenarma

#endif
