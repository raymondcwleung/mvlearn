#include "metrics/pairwise.h"

#include <Eigen/Dense>
#include <cmath>
#include <mlpack.hpp>

#include "neighbors/nearestneighbors.h"
#include "utils_eigenarma/conversions.h"

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

  for (int i = 0; i < X.rows(); i++) {
    for (int j = 0; j < Y.rows(); j++) {
      Kmat(i, j) = std::exp(
          -1.0 * gamma * (X(i, Eigen::all) - Y(j, Eigen::all)).squaredNorm());
    }
  }

  return Kmat;
}

Eigen::MatrixXd rbfKernel(const Eigen::Ref<const Eigen::MatrixXd>& X,
                          double gamma) {
  return rbfKernel(X, X, gamma);
}

//! Compute a local scaling RBF kernel between the rows of X and the rows of Y.
/*!
 * See Zelnik-Manor and Perona (2004).
 */
Eigen::MatrixXd rbfLocalKernel(const Eigen::Ref<const Eigen::MatrixXd>& X,
                               int num_neighbors) {
  // Number of rows of X is the number of sample points. Number of columns of X
  // is the dimension size of the vector space.
  int num_samples = X.rows();
  Eigen::MatrixXd Kmat(num_samples, num_samples);

  // Find the K-th neighbor of each point x_i. Let x_K be that K-th neighbor
  // of x_i. Then the local scale for the point x_i is \gamma_i = d(x_i, x_K).
  // Take the Eigen based matrix data and convert it to an Armadillo matrix
  arma::Mat<double> arma_X = utilseigenarma::castEigenToArma<double>(X);
  // Critical to take the transpose for mlpack::NeighborSearch
  arma_X = arma_X.t();

  mlpack::NeighborSearch<mlpack::NearestNeighborSort, mlpack::EuclideanDistance>
      nn(arma_X);
  arma::Mat<std::size_t> arma_neighbors;
  arma::Mat<double> arma_distances;
  nn.Search(num_neighbors, arma_neighbors, arma_distances);

  // distances is num_features x num_samples matrix
  Eigen::MatrixXd distances = utilseigenarma::castArmaToEigen(arma_distances);

  // Extract the K-th neighbor distances for each of the points x_i
  Eigen::VectorXd local_scales = distances.transpose()(Eigen::all, Eigen::last);

  double scaling{0.0};
  for (int i = 0; i < num_samples; i++) {
    for (int j = 0; j < num_samples; j++) {
      if (i == j) {
        Kmat(i, j) = 0.0;
      } else {
        scaling = local_scales(i) * local_scales(j);
        Kmat(i, j) = std::exp(
            -1.0 * (X(i, Eigen::all) - X(j, Eigen::all)).squaredNorm() /
            scaling);
      }
    }
  }

  return Kmat;
}

}  // namespace sklearncpp::metrics::pairwise
