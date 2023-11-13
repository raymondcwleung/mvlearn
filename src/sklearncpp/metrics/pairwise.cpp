#include "metrics/pairwise.h"

#include <Eigen/Dense>
#include <cmath>

namespace sklearncpp::metrics::pairwise {

//! Compute the RBF (Gaussian / Radial basis function) kernel between the rows
//! of X and the rows of Y.
/*!
 * \param X An \f$ n \times m \f$ matrix, where \f$ n \f$ is the number of
 * samples and \f$ m \f$ is the number of features.
 *
 * \param Y An \f$ n \times m
 * \f$ matrix, where \f$ n \f$ is the number of samples and \f$ m \f$ is the
 * number of features.
 *
 * \param gamma Float
 *
 * \return Return a matrix of shape \f$ n \times n \f$.
 */
Eigen::MatrixXd rbfKernel(const Eigen::Ref<const Eigen::MatrixXd>& X,
                          const Eigen::Ref<const Eigen::MatrixXd>& Y,
                          double gamma) {
  // NOTE:: We're not checking that X and Y have the same shapes
  Eigen::MatrixXd Kmat(X.rows(), X.rows());

  for (int i; i < X.rows(); i++) {
    for (int j; j < Y.rows(); j++) {
      Kmat(i, j) = std::exp(
          -1.0 * gamma * (X(i, Eigen::all) - Y(j, Eigen::all)).squaredNorm());
    }
  }

  return Kmat;
}

}  // namespace sklearncpp::metrics::pairwise
