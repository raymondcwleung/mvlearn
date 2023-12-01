#include "cluster/mv_spectral.h"

#include <pstl/glue_execution_defs.h>

#include <Eigen/Dense>

#include "cluster/ClusterRotate.h"

/* #include <Eigen/StdVector> */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <execution>
#include <functional>
/* #include <map> */
#include <mlpack.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <numeric>
#include <string>
#include <utility>

#include "Eigen/src/Core/Matrix.h"
#include "scipycpp/spatial/distance/distance.h"
#include "sklearncpp/cluster/kmeans.h"
#include "sklearncpp/metrics/pairwise.h"
#include "sklearncpp/neighbors/nearestneighbors.h"
#include "utils_eigenarma/conversions.h"

namespace mvlearn::cluster {

MVSpectralClustering::MVSpectralClustering(
    int n_clusters, int num_samples, int num_features, int random_state,
    int info_view, int max_iter, int n_init, std::string affinity,
    int n_neighbors, double gamma, bool local_gamma, bool auto_num_clusters)
    : n_clusters_(n_clusters),
      num_samples_{num_samples},
      num_features_{num_features},
      random_state_{random_state},
      info_view_{info_view},
      max_iter_{max_iter},
      n_init_(n_init),
      affinity_{affinity},
      gamma_{gamma},
      n_neighbors_{n_neighbors},
      local_gamma_{local_gamma},
      auto_num_clusters_(auto_num_clusters) {
  // To ensure correct sizes
  embedding_.resize(num_samples_, n_clusters_);
}

// Computes the affinity matrix based on the selected kernel type
Eigen::MatrixXd MVSpectralClustering::affinityMat_(const Eigen::MatrixXd& X) {
  // Produce the affinity matrix based on the selected kernel type
  Eigen::MatrixXd sims(num_samples_, num_samples_);

  if (affinity_ == "rbf_constant_scale") {
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
    sims = sklearncpp::metrics::pairwise::rbfKernel(X, X, gamma);

  } else if (affinity_ == "rbf_local_scale") {
    sims = sklearncpp::metrics::pairwise::rbfLocalKernel(X);

  } else if (affinity_ == "nearest_neighbors") {
    sims = sklearncpp::neighbors::nearestNeighbors<mlpack::NearestNeighborSort,
                                                   mlpack::EuclideanDistance>(
               X, n_neighbors_, num_samples_)
               .cast<double>();
  } else if (affinity_ == "nearest_neighbors_l1") {
    sims = sklearncpp::neighbors::nearestNeighbors<mlpack::NearestNeighborSort,
                                                   mlpack::ManhattanDistance>(
               X, n_neighbors_, num_samples_)
               .cast<double>();
  } else {
    // TODO
    // sdfhs
  }

  return sims;
}

Eigen::MatrixXd MVSpectralClustering::computeEigs_(
    const Eigen::Ref<const Eigen::MatrixXd>& X) {
  // Compute the normalized Laplacian
  Eigen::VectorXd v_alt = X.colwise().sum().cwiseInverse().cwiseSqrt();
  Eigen::MatrixXd laplacian = v_alt.asDiagonal() * X * v_alt.asDiagonal();

  // Make the resulting matrix symmetric
  laplacian = 0.5 * (laplacian + laplacian.transpose());

  // To ensure PSD
  double min_val = laplacian.array().minCoeff();
  if (min_val < 0.0) {
    laplacian =
        laplacian + Eigen::MatrixXd::Constant(laplacian.rows(),
                                              laplacian.cols(), -1.0 * min_val);
  }

  // Obtain the top n_cluster eigenvectors the of the Laplacian
  // Note Eigen::SelfAdjointEigenSolver sorts the eigenvalues in increasing
  // order
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(laplacian);

  Eigen::MatrixXd la_eigs;
  if (n_clusters_ != 0) {
    la_eigs = es.eigenvectors()(
        Eigen::all, Eigen::seq(laplacian.cols() - n_clusters_, Eigen::last));
  } else {
    la_eigs = es.eigenvectors();
  }

  /* // HACK */
  /* ClusterRotate clusterrotate{1}; */
  /* std::vector<std::vector<int>> clusters = clusterrotate.cluster(la_eigs); */
  /* int mynumclusters = clusters.size(); */
  /**/
  /* std::cout << "mynumclusters: " << mynumclusters << "\n"; */

  return la_eigs;
}

//! Performs clustering on the multiple views of data
/*
 *
 */
void MVSpectralClustering::fit(const std::vector<Eigen::MatrixXd>& Xs) {
  n_views_ = Xs.size();

  // Compute the similarity matrices W_v for each of the views
  // The affinity matrix for each view is of size num_samples_ x num_samples_
  std::vector<Eigen::MatrixXd> sims(
      n_views_, Eigen::MatrixXd(num_samples_, num_samples_));
  std::transform(std::execution::par_unseq, Xs.begin(), Xs.end(), sims.begin(),
                 [this, &Xs = std::as_const(Xs)](const Eigen::MatrixXd& X) {
                   return affinityMat_(X);
                 });

  // Initialize matrices of eigenvectors U_v for each view
  // The matrix of top eigenvectors are of size num_samples_ x n_clusters_
  std::vector<Eigen::MatrixXd> U_mats(
      n_views_, Eigen::MatrixXd(num_samples_, n_clusters_));
  std::transform(std::execution::par_unseq, sims.begin(), sims.end(),
                 U_mats.begin(),
                 [this, &sims = std::as_const(sims)](const Eigen::MatrixXd& X) {
                   return computeEigs_(X);
                 });

  // Iteratively compute new graph similarities, Laplacians and eigenvectors
  std::vector<Eigen::MatrixXd> eig_sums(
      n_views_, Eigen::MatrixXd(num_samples_, num_samples_));
  std::vector<Eigen::MatrixXd> new_sims(
      n_views_, Eigen::MatrixXd(num_samples_, num_samples_));
  std::vector<Eigen::MatrixXd> mat1(
      n_views_, Eigen::MatrixXd(num_samples_, num_samples_));

  Eigen::MatrixXd U_sum(num_samples_, num_samples_);
  std::vector<int> idx_views(n_views_);
  std::iota(idx_views.begin(), idx_views.end(), 0);
  int iter{0};
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

    // Compute new graph similariity representation S_v
    std::iota(idx_views.begin(), idx_views.end(), 0);
    std::for_each(
        std::execution::par_unseq, idx_views.begin(), idx_views.end(),
        [&U_sum = std::as_const(U_sum), &eig_sums = std::as_const(eig_sums),
         &sims = std::as_const(sims), &new_sims, &mat1](const int& view) {
          mat1[view] = sims[view] * (U_sum - eig_sums[view]);
          new_sims[view] = 0.5 * (mat1[view] + mat1[view].transpose());
        });

    // Recompute eigenvectors and get new U_v's
    std::transform(std::execution::par_unseq, new_sims.begin(), new_sims.end(),
                   U_mats.begin(),
                   [this, &new_sims = std::as_const(new_sims)](
                       const Eigen::MatrixXd& X) { return computeEigs_(X); });

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
