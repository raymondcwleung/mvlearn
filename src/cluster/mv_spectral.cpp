#include "cluster/mv_spectral.h"

#include <Eigen/Dense>
#include <armadillo>
#include <cmath>
#include <string>

#include "metrics/pairwise/pairwise.h"
#include "spatial/distance/distance.h"
#include "utils/utils_eigen.cpp"

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
  // value for this view.
  double gamma{};
  if (gamma_ == -1) {
    Eigen::MatrixXd distances = spatial::distance::cdist(X, X);

    // Compute the median of the distances matrix
    arma::Mat<double> arma_X = utils::utilseigen::castEigenToArma(X);
    arma::Col<double> arma_vecX = arma::vectorise(arma_X);
    double median = arma::median(arma_vecX);

    gamma = 1.0 / (2.0 * std::pow(median, 2));
  } else {
    gamma = gamma_;
  }

  // Produce the affinity matrix based on the selected kernel type
  Eigen::MatrixXd sims{};
  if (affinity_ == "rbf") {
    sims = metrics::pairwise::rbfKernel(X, X, gamma);
  }

  return sims;
}

}  // namespace mvlearn::cluster
