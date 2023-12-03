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
#include <stdexcept>
#include <string>
#include <utility>

#include "Eigen/src/Core/Matrix.h"
#include "scipycpp/spatial/distance/distance.h"
#include "sklearncpp/cluster/kmeans.h"
#include "sklearncpp/metrics/pairwise.h"
#include "sklearncpp/neighbors/nearestneighbors.h"
#include "utils_eigenarma/conversions.h"

namespace mvlearn::cluster {

MVSpectralClustering::MVSpectralClustering(int n_clusters, int num_samples,
                                           int num_features, int random_state,
                                           int info_view, int max_iter,
                                           int n_init, std::string affinity,
                                           int n_neighbors, double gamma,
                                           bool auto_num_clusters)
    : n_clusters_(n_clusters),
      num_samples_{num_samples},
      num_features_{num_features},
      random_state_{random_state},
      info_view_{info_view},
      max_iter_{max_iter},
      n_init_(n_init),
      affinity_{affinity},
      n_neighbors_{n_neighbors},
      gamma_{gamma},
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
    throw std::invalid_argument("Invalid affinity choice");
  }

  return sims;
}

Eigen::MatrixXd MVSpectralClustering::constructLaplacian_(
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

  return laplacian;
}

Eigen::MatrixXd MVSpectralClustering::computeEigs_(
    const Eigen::Ref<const Eigen::MatrixXd>& X, int num_top_eigenvectors) {
  // Get the normalized Laplacian
  Eigen::MatrixXd laplacian = constructLaplacian_(X);

  // Obtain the top n_cluster eigenvectors the of the Laplacian
  // Note Eigen::SelfAdjointEigenSolver sorts the eigenvalues in increasing
  // order
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(laplacian);

  Eigen::MatrixXd la_eigs;
  if (num_top_eigenvectors == -1) {
    la_eigs = es.eigenvectors();
  } else {
    la_eigs = es.eigenvectors()(
        Eigen::all,
        Eigen::seq(laplacian.cols() - num_top_eigenvectors, Eigen::last));
  }

  return la_eigs;
}

void MVSpectralClustering::computeEigs_(
    // inputs
    const Eigen::Ref<const Eigen::MatrixXd>& X, int num_top_eigenvectors,
    // outputs
    Eigen::MatrixXd& u_mat, Eigen::MatrixXd& laplacian, double& obj_val) {
  // Get the normalized Laplacian
  laplacian = constructLaplacian_(X);

  // Obtain the top n_cluster eigenvectors the of the Laplacian
  // Note Eigen::SelfAdjointEigenSolver sorts the eigenvalues in increasing
  // order
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(laplacian);

  Eigen::VectorXd eigenvals;
  if (num_top_eigenvectors == -1) {
    u_mat = es.eigenvectors();
    eigenvals = es.eigenvalues();
  } else {
    u_mat = es.eigenvectors()(
        Eigen::all,
        Eigen::seq(laplacian.cols() - num_top_eigenvectors, Eigen::last));
    eigenvals = es.eigenvalues()(
        Eigen::seq(laplacian.cols() - num_top_eigenvectors, Eigen::last));
  }

  obj_val = eigenvals.sum();
}

//! Setup the initial affinity matrices at the beginning of the
//! fitting procedure
void MVSpectralClustering::fit_init_(const std::vector<Eigen::MatrixXd>& Xs,
                                     std::vector<Eigen::MatrixXd>& sims,
                                     int& num_clusters_info_view) {
  // Compute the initial affinity matrices
  std::transform(std::execution::par_unseq, Xs.begin(), Xs.end(), sims.begin(),
                 [this, &Xs = std::as_const(Xs)](const Eigen::MatrixXd& X) {
                   return affinityMat_(X);
                 });

  // Compute the eigendecomposition of the Laplacian using only the most
  // informative view
  if (auto_num_clusters_) {
    Eigen::MatrixXd la_eigs_info_view =
        computeEigs_(sims[info_view_], n_clusters_);

    // Apply the Zelnik-Manor and Perona (2004) method to compute the
    // number of clusters based on the data of the most informative
    // view
    ClusterRotate clusterrotate{};
    std::vector<std::vector<int>> clusters =
        clusterrotate.cluster(la_eigs_info_view);
    num_clusters_info_view = clusters.size();

    n_clusters_ = num_clusters_info_view;
  }
};

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
  int num_clusters_info_view{};
  fit_init_(Xs, sims, num_clusters_info_view);

  n_clusters_ = num_clusters_info_view;

  // Initialize matrices of eigenvectors U_v for each view
  // The matrix of top eigenvectors are of size num_samples_ x n_clusters_
  std::vector<Eigen::MatrixXd> U_mats(
      n_views_, Eigen::MatrixXd(num_samples_, n_clusters_));
  std::transform(std::execution::par_unseq, sims.begin(), sims.end(),
                 U_mats.begin(),
                 [this, &sims = std::as_const(sims)](const Eigen::MatrixXd& X) {
                   return computeEigs_(X, n_clusters_);
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
    std::transform(
        std::execution::par_unseq, new_sims.begin(), new_sims.end(),
        U_mats.begin(),
        [this, &new_sims = std::as_const(new_sims)](const Eigen::MatrixXd& X) {
          return computeEigs_(X, n_clusters_);
        });

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

int mvlearn::cluster::MVSpectralClustering::get_num_clusters() {
  return n_clusters_;
}

}  // namespace mvlearn::cluster
