#include "cluster/mv_spectral.h"

#include <Eigen/Dense>
#include <armadillo>
#include <cmath>
#include <mlpack.hpp>
#include <string>

/* #include "metrics/pairwise/pairwise.h" */
#include "mlpack/core/kernels/gaussian_kernel.hpp"
#include "mlpack/core/metrics/lmetric.hpp"
#include "mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp"
#include "scipycpp/spatial/distance/distance.h"
#include "sklearncpp/metrics/pairwise.h"
#include "sklearncpp/neighbors/nearestneighbors.h"
#include "utils_eigenarma/conversions.h"

namespace mvlearn::cluster {
MVSpectralClustering::MVSpectralClustering(int n_clusters, int random_state,
                                           int info_view, int max_iter,
                                           int n_init, std::string affinity,
                                           int n_neighbors, float gamma = -1)
    : n_clusters_(n_clusters),
      random_state_{random_state},
      info_view_{info_view},
      max_iter_{max_iter},
      n_init_(n_init),
      affinity_{affinity},
      gamma_{gamma},
      n_neighbors_{n_neighbors} {}

// Computes the affinity matrix based on the selected kernel type
Eigen::MatrixXd MVSpectralClustering::affinityMat_(
    const Eigen::Ref<const Eigen::MatrixXd>& X) {
  // A gamma has not been provided. Compute a gamma
  // value for this view. Note the gamma parameter is interpretted
  // as a bandwidth parameter.
  double gamma{};
  if (gamma_ == -1) {
    Eigen::MatrixXd distances = scipycpp::spatial::distance::cdist(X, X);

    // Compute the median of the distances matrix
    arma::Mat<double> arma_X = utilseigenarma::castEigenToArma<double>(X);
    arma::Col<double> arma_vecX = arma::vectorise(arma_X);
    double median = arma::median(arma_vecX);

    gamma = 1.0 / (2.0 * std::pow(median, 2));
  } else {
    gamma = gamma_;
  }

  // Produce the affinity matrix based on the selected kernel type
  Eigen::MatrixXd sims{};

  if (affinity_ == "rbf") {
    sims = sklearncpp::metrics::pairwise::rbfKernel(X, X, gamma);

  } else if (affinity_ == "nearest_neighbors") {
    sims = sklearncpp::neighbors::nearestNeighbors<mlpack::NearestNeighborSort,
                                                   mlpack::EuclideanDistance>(
               X, n_neighbors_)
               .cast<double>();

  } else {
    // TODO
  }

  return sims;
}

Eigen::MatrixXd MVSpectralClustering::computeEigs_(
    const Eigen::Ref<const Eigen::MatrixXd>& X) {
  // Compute the normalized Laplacian
  Eigen::VectorXd col_sums = X.colwise().sum();
  Eigen::MatrixXd d_mat = Eigen::MatrixXd(X.colwise().sum().asDiagonal());
  Eigen::MatrixXd d_alt = d_mat.inverse().cwiseSqrt();
  Eigen::MatrixXd laplacian = d_alt * X * d_alt;

  // Make the resulting matrix symmetric
  laplacian = (laplacian + laplacian.transpose()) / 2.0;

  // To ensure PSD
  double min_val = laplacian.minCoeff();
  if (min_val < 0.0) {
    laplacian = laplacian + Eigen::MatrixXd::Constant(
                                laplacian.rows(), laplacian.cols(), min_val);
  }

  // Obtain the top n_cluster eigenvectors the of the Laplacian
  // Note Eigen::SelfAdjointEigenSolver sorts the eigenvalues in increasing
  // order
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(laplacian);
  Eigen::MatrixXd u_mat{es.eigenvectors()};

  Eigen::MatrixXd la_eigs =
      u_mat(Eigen::all, Eigen::seq(u_mat.cols() - n_clusters_, Eigen::last));

  return la_eigs;
}

}  // namespace mvlearn::cluster
