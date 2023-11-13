#include "spatial/distance/distance.h"

#include <Eigen/Dense>
#include <string>

namespace scipycpp::spatial::distance {

//! An implementation of scipy.spatial.distance.cdist
/*!
 * \brief Compute distance between each pair of the two collections of inputs.
 *
 * \param XA An \f$m \times n\f$ matrix of \f$ m \f$ original observations
 * in an \f$ n \f$-dimensional space.
 * \param XB An \f$m \times n\f$ matrix of \f$ m \f$ original observations
 * in an \f$ n \f$-dimensional space.
 * \param metric The distance metric to use.
 *
 * \return A \f$ m \times m \f$ distance matrix \f$ Y \f$ is returned.
 */
template <typename Derived>
Eigen::MatrixXd cdist(const Eigen::MatrixBase<Derived>& XA,
                      const Eigen::MatrixBase<Derived>& XB,
                      const std::string metric) {
  // NOTE:: Not checking that XA.rows() is the same as XB.rows()
  int num_features = static_cast<int>(XA.rows());

  // NOTE:: Not checking that XA.cols() is the same as XB.cols()
  int num_samples = static_cast<int>(XA.cols());

  Eigen::MatrixXd dist_mat(num_features, num_features);

  Eigen::VectorXd vec(num_samples);

  for (int i = 0; i < num_features; i++) {
    for (int j = 0; j < num_features; j++) {
      vec = XA(i, Eigen::all) - XB(j, Eigen::all);
      dist_mat(i, j) = vec.norm();
    }
  }

  return dist_mat;
}

}  // namespace scipycpp::spatial::distance
