#include "cluster/mv_spectral.h"

#include <pstl/glue_execution_defs.h>

#include <Eigen/Dense>
/* #include <Eigen/StdVector> */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <execution>
#include <functional>
/* #include <map> */
#include <mlpack.hpp>
#include <numeric>
#include <string>

#include "Eigen/src/Core/Matrix.h"
#include "scipycpp/spatial/distance/distance.h"
#include "sklearncpp/cluster/kmeans.h"
#include "sklearncpp/metrics/pairwise.h"
#include "sklearncpp/neighbors/nearestneighbors.h"
#include "utils_eigenarma/conversions.h"

namespace mvlearn::cluster {
typedef std::chrono::high_resolution_clock Clock;

MVSpectralClustering::MVSpectralClustering(int n_clusters, int num_samples,
                                           int num_features, int random_state,
                                           int info_view, int max_iter,
                                           int n_init, std::string affinity,
                                           int n_neighbors, double gamma)
    : n_clusters_(n_clusters),
      num_samples_{num_samples},
      num_features_{num_features},
      random_state_{random_state},
      info_view_{info_view},
      max_iter_{max_iter},
      n_init_(n_init),
      affinity_{affinity},
      gamma_{gamma},
      n_neighbors_{n_neighbors} {
  // To ensure correct sizes
  embedding_.resize(num_samples_, n_clusters_);
}

// Computes the affinity matrix based on the selected kernel type
Eigen::MatrixXd MVSpectralClustering::affinityMat_(const Eigen::MatrixXd& X) {
  // A gamma has not been provided. Compute a gamma
  // value for this view. Note the gamma parameter is interpretted
  // as a bandwidth parameter.
  double gamma{};
  if (gamma_ == -1) {
    Eigen::MatrixXd distances = scipycpp::spatial::distance::cdist(X, X);

    // Vectorize the distances matrix
    Eigen::VectorXd v_dist = distances.reshaped();
    std::vector<double> vec_distances;
    vec_distances.resize(v_dist.size());
    Eigen::VectorXd::Map(&vec_distances[0], v_dist.size()) = v_dist;

    // Compute the median of the distances
    size_t midpt = vec_distances.size() / 2;
    std::nth_element(vec_distances.begin(), vec_distances.begin() + midpt,
                     vec_distances.end());
    double median = vec_distances[midpt];

    gamma = 1.0 / (2.0 * std::pow(median, 2));
  } else {
    gamma = gamma_;
  }

  // Produce the affinity matrix based on the selected kernel type
  /* Eigen::MatrixXd sims(num_samples_, num_samples_); */
  Eigen::MatrixXd sims;

  if (affinity_ == "rbf") {
    sims = sklearncpp::metrics::pairwise::rbfKernel(X, X, gamma);
  }
  /* else if (affinity_ == "nearest_neighbors") { */
  /*   sims =
   * sklearncpp::neighbors::nearestNeighbors<mlpack::NearestNeighborSort, */
  /*                                                  mlpack::EuclideanDistance>(
   */
  /*              X, n_neighbors_) */
  /*              .cast<double>(); */
  /**/
  /* } else { */
  /*   // TODO */
  /* } */

  return sims;
}

Eigen::MatrixXd MVSpectralClustering::computeEigs_(
    const Eigen::Ref<const Eigen::MatrixXd>& X) {
  // Compute the normalized Laplacian
  Eigen::VectorXd col_sums = X.colwise().sum();

  Eigen::VectorXd v_alt = X.colwise().sum().cwiseInverse().cwiseSqrt();
  Eigen::MatrixXd laplacian = v_alt.asDiagonal() * X * v_alt.asDiagonal();

  // Make the resulting matrix symmetric
  laplacian = 0.5 * (laplacian + laplacian.transpose());

  // To ensure PSD
  /* double min_val = laplacian.array().minCoeff(); */
  /* if (min_val < 0.0) { */
  /*   laplacian = laplacian + Eigen::MatrixXd::Constant( */
  /*                               laplacian.rows(), laplacian.cols(), min_val);
   */
  /* } */

  // Obtain the top n_cluster eigenvectors the of the Laplacian
  // Note Eigen::SelfAdjointEigenSolver sorts the eigenvalues in increasing
  // order
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(laplacian);

  return es.eigenvectors()(
      Eigen::all, Eigen::seq(laplacian.cols() - n_clusters_, Eigen::last));
}

//! Performs clustering on the multiple views of data
/*
 *
 */
void MVSpectralClustering::fit(const std::vector<Eigen::MatrixXd>& Xs) {
  n_views_ = Xs.size();

  // Compute the similarity matrices
  std::vector<Eigen::MatrixXd> sims(n_views_);
  std::for_each(sims.begin(), sims.end(), [this](Eigen::MatrixXd& X) {
    X.resize(num_samples_, num_features_);
  });
  std::transform(std::execution::par_unseq, Xs.begin(), Xs.end(), sims.begin(),
                 [&](const Eigen::MatrixXd& X) { return affinityMat_(X); });

  // Initialize matrices of eigenvectors
  std::vector<Eigen::MatrixXd> U_mats(n_views_);
  std::for_each(U_mats.begin(), U_mats.end(), [this](Eigen::MatrixXd& X) {
    X.resize(num_samples_, n_clusters_);
  });
  std::transform(std::execution::par_unseq, sims.begin(), sims.end(),
                 U_mats.begin(),
                 [&](const Eigen::MatrixXd& X) { return computeEigs_(X); });

  // Iteratively compute new graph similarities, Laplacians and eigenvectors
  int iter{0};

  std::vector<Eigen::MatrixXd> eig_sums(n_views_);
  std::for_each(eig_sums.begin(), eig_sums.end(), [this](Eigen::MatrixXd& X) {
    X.resize(num_samples_, num_features_);
  });

  std::vector<Eigen::MatrixXd> new_sims(n_views_);
  std::for_each(new_sims.begin(), new_sims.end(), [this](Eigen::MatrixXd& X) {
    X.resize(num_samples_, num_features_);
  });

  Eigen::MatrixXd U_sum(num_samples_, num_samples_);

  std::vector<int> idx_views(n_views_);
  std::iota(idx_views.begin(), idx_views.end(), 0);

  while (iter < max_iter_) {
    // Compute the sums of the products of the spectral embeddings and their
    // transposes.
    // Note that each u_mat is of size num_samples x n_cluster. Hence,
    // each entry in eig_sums is num_samples x num_samples.
    std::transform(std::execution::par_unseq, U_mats.begin(), U_mats.end(),
                   eig_sums.begin(), [&](const Eigen::MatrixXd& u_mat) {
                     return u_mat * u_mat.transpose();
                   });

    U_sum.setZero();
    std::for_each(eig_sums.begin(), eig_sums.end(),
                  [&U_sum](const Eigen::MatrixXd& X) { return U_sum += X; });

    // Compute new graph similariity representation
    std::iota(idx_views.begin(), idx_views.end(), 0);
    std::for_each(std::execution::par_unseq, idx_views.begin(), idx_views.end(),
                  [&U_sum, &eig_sums, &new_sims, &sims](const int& view) {
                    Eigen::MatrixXd mat1 =
                        sims[view] * (U_sum - eig_sums[view]);
                    new_sims[view] = 0.5 * (mat1 + mat1.transpose());
                  });

    // Recompute eigenvectors
    std::transform(std::execution::par_unseq, new_sims.begin(), new_sims.end(),
                   U_mats.begin(),
                   [&](const Eigen::MatrixXd& X) { return computeEigs_(X); });

    iter++;
  }

  // Row normalize
  for (int view = 0; view < n_views_; view++) {
    for (int j = 0; j < U_mats[view].rows(); j++) {
      U_mats[view].row(j).normalize();
    }
  }

  // Perform k-means clustering
  sklearn::cluster::KMeans kmeans(n_clusters_, max_iter_);
  embedding_ = U_mats[info_view_];
  labels_ = kmeans.fit_predict(embedding_);
}

Eigen::VectorXi mvlearn::cluster::MVSpectralClustering::fit_predict(
    const std::vector<Eigen::MatrixXd>& Xs) {
  fit(Xs);

  return labels_;
}

}  // namespace mvlearn::cluster
