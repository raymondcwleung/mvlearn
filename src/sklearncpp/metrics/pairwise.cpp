#include "metrics/pairwise.h"

#include <omp.h>

#include <Eigen/Dense>
#include <cmath>
#include <mlpack.hpp>

#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Core/util/Constants.h"
#include "mlpack/core/kernels/cosine_distance.hpp"
#include "mlpack/core/kernels/gaussian_kernel.hpp"
#include "mlpack/core/kernels/linear_kernel.hpp"
#include "mlpack/core/kernels/spherical_kernel.hpp"
#include "mlpack/core/metrics/ip_metric.hpp"
#include "mlpack/core/metrics/lmetric.hpp"
#include "mlpack/core/metrics/mahalanobis_distance.hpp"
#include "mlpack/core/tree/binary_space_tree/typedef.hpp"
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

Eigen::MatrixXd cosineKernel(const Eigen::Ref<const Eigen::MatrixXd>& X) {
  int num_samples = X.rows();

  arma::Mat<double> arma_X = utilseigenarma::castEigenToArma<double>(X);
  arma::Mat<double> arma_Kmat(num_samples, num_samples);

  std::cout << "arma_X nrow" << arma_X.n_rows << "\n";
  std::cout << "arma_X ncol" << arma_X.n_cols << "\n";
  /* std::cout << "arma_X" << arma_X << "\n"; */

  mlpack::metric::CosineDistance dist;

  for (int i = 0; i < num_samples; i++) {
    for (int j = 0; j < num_samples; j++) {
      arma_Kmat(i, j) = dist.Evaluate(arma_X.row(i), arma_X.row(j));
    }
  }

  std::cout << "Done calc"
            << "\n";

  Eigen::MatrixXd Kmat = utilseigenarma::castArmaToEigen<double>(arma_Kmat);

  return Kmat;
}

//! Compute a local scaling RBF kernel between the rows of X and the rows of Y.
/*!
 * See Zelnik-Manor and Perona (2004).
 */
Eigen::MatrixXd rbfLocalKernel(const Eigen::Ref<const Eigen::MatrixXd>& X,
                               int norm_p,
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

  arma::Mat<std::size_t> arma_neighbors;
  arma::Mat<double> arma_distances;

  if (norm_p == 2) {
    mlpack::NeighborSearch<mlpack::NearestNeighborSort,  // SortPolicy
                           mlpack::EuclideanDistance,    // MetricType
                           arma::mat,                    // MatType
                           mlpack::tree::VPTree          // TreeType
                           >
        nn(arma_X);
    nn.Search(num_neighbors, arma_neighbors, arma_distances);
  } else if (norm_p == 1) {
    mlpack::NeighborSearch<mlpack::NearestNeighborSort,  // SortPolicy
                           mlpack::ManhattanDistance,    // MetricType
                           arma::mat,                    // MatType
                           mlpack::tree::VPTree          // TreeType
                           >
        nn(arma_X);
    nn.Search(num_neighbors, arma_neighbors, arma_distances);
  }

  // distances is num_features x num_samples matrix
  Eigen::MatrixXd distances = utilseigenarma::castArmaToEigen(arma_distances);

  // Extract the K-th neighbor distances for each of the points x_i
  Eigen::VectorXd local_scales = distances.transpose()(Eigen::all, Eigen::last);

#pragma omp parallel for
  for (int i = 0; i < num_samples; i++) {
    for (int j = 0; j < num_samples; j++) {
      if (i == j) {
        Kmat(i, j) = 0.0;
      } else {
        double scaling = local_scales(i) * local_scales(j);

        double norm_dist_val;
        if (norm_p == 2) {
          norm_dist_val = (X(i, Eigen::all) - X(j, Eigen::all)).lpNorm<2>();
        } else if (norm_p == 1) {
          norm_dist_val = (X(i, Eigen::all) - X(j, Eigen::all)).lpNorm<1>();
        }

        Kmat(i, j) = std::exp(-1.0 * norm_dist_val / scaling);
      }
    }
  }

  return Kmat;
}

}  // namespace sklearncpp::metrics::pairwise
