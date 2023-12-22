#include "cluster/mv_spectral.h"

#include <Eigen/src/Core/Matrix.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/Util/CompInfo.h>
#include <Spectra/Util/SelectionRule.h>
#include <pstl/glue_execution_defs.h>

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <execution>
#include <functional>
#include <stdexcept>

#include "cluster/ClusterRotate.h"
/* #include <map> */
#include <chrono>
#include <mlpack.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "scipycpp/spatial/distance/distance.h"
#include "sklearncpp/cluster/kmeans.h"
#include "sklearncpp/metrics/pairwise.h"
#include "sklearncpp/neighbors/nearestneighbors.h"
#include "utils_eigenarma/conversions.h"

namespace mvlearn::cluster {

MVSpectralClustering::MVSpectralClustering(int n_clusters,
                                           int num_samples,
                                           int num_features,
                                           int info_view,
                                           int max_iter,
                                           std::string affinity,
                                           int n_neighbors,
                                           double gamma,
                                           bool auto_num_clusters,
                                           bool use_spectra)
    : n_clusters_(n_clusters),
      num_samples_{num_samples},
      num_features_{num_features},
      info_view_{info_view},
      max_iter_{max_iter},
      affinity_{affinity},
      n_neighbors_{n_neighbors},
      gamma_{gamma},
      auto_num_clusters_(auto_num_clusters),
      use_spectra_(use_spectra) {
  // To ensure correct sizes
  embedding_.resize(num_samples_, n_clusters_);
}

// Computes the affinity matrix based on the selected kernel type
Eigen::MatrixXd MVSpectralClustering::affinityMat_(
    const Eigen::Ref<const Eigen::MatrixXd>& X) {
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
      std::nth_element(vec_distances.begin(),
                       vec_distances.begin() + midpt,
                       vec_distances.end());
      double median = vec_distances[midpt];

      gamma = 1.0 / (2.0 * std::pow(median, 2));
    } else {
      gamma = gamma_;
    }
    sims = sklearncpp::metrics::pairwise::rbfKernel(X, X, gamma);

  } else if (affinity_ == "cosine") {
    sims = sklearncpp::metrics::pairwise::cosineKernel(X);

    std::cout << sims << "\n";

  } else if (affinity_ == "rbf_local_scale_l2") {
    sims = sklearncpp::metrics::pairwise::rbfLocalKernel(X, 2);

  } else if (affinity_ == "rbf_local_scale_l1") {
    sims = sklearncpp::metrics::pairwise::rbfLocalKernel(X, 1);

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

void MVSpectralClustering::constructLaplacian_(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    Eigen::MatrixXd& laplacian) {
  // Compute the normalized Laplacian
  Eigen::VectorXd v_alt(num_samples_);
  v_alt.noalias() = X.colwise().sum().cwiseInverse().cwiseSqrt();
  laplacian.noalias() = v_alt.asDiagonal() * X * v_alt.asDiagonal();
  laplacian = 0.5 * (laplacian + laplacian.transpose());
}

Eigen::MatrixXd MVSpectralClustering::constructLaplacian_(
    const Eigen::Ref<const Eigen::MatrixXd>& X) {
  // Compute the normalized Laplacian
  Eigen::VectorXd v_alt = X.colwise().sum().cwiseInverse().cwiseSqrt();
  Eigen::MatrixXd laplacian = v_alt.asDiagonal() * X * v_alt.asDiagonal();

  return (0.5 * laplacian + 0.5 * laplacian.transpose());
}

void MVSpectralClustering::computeEigs_(
    // inputs
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    int num_top_eigenvectors,
    // outputs
    Eigen::MatrixXd& u_mat,      // Matrix of the top eigenvectors
    Eigen::MatrixXd& laplacian,  // Laplacian matrix
    double& obj_val              // Sum of eigenvalues
) {
  // Get the normalized Laplacian
  laplacian.noalias() = constructLaplacian_(X);

  if (use_spectra_) {
    // Try to use the Spectra package to compute the eigenvectors for
    // efficiency. If Spectra fails, fall back to the the eigendecomposition
    // solver in Eigen.
    Spectra::DenseSymMatProd<double> op(laplacian);
    Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(
        op,                       // op
        num_top_eigenvectors,     // nev
        2 * num_top_eigenvectors  // ncv
    );
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestAlge);

    if (eigs.info() == Spectra::CompInfo::Successful) {
      u_mat.noalias() = eigs.eigenvectors();
      obj_val = eigs.eigenvalues().sum();

    } else {
      // Note Eigen::SelfAdjointEigenSolver sorts the eigenvalues in increasing
      // order
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(laplacian);

      u_mat.noalias() = es.eigenvectors()(
          Eigen::all,
          Eigen::seq(laplacian.cols() - num_top_eigenvectors, Eigen::last));
      obj_val = es.eigenvalues().sum();
    }
  } else {
    // Note Eigen::SelfAdjointEigenSolver sorts the eigenvalues in increasing
    // order
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(laplacian);

    u_mat.noalias() = es.eigenvectors()(
        Eigen::all,
        Eigen::seq(laplacian.cols() - num_top_eigenvectors, Eigen::last));
    obj_val = es.eigenvalues().sum();
  }
}

//! Setup the initial affinity matrices at the beginning of the
//! fitting procedure
void MVSpectralClustering::fit_init_(const std::vector<Eigen::MatrixXd>& Xs,
                                     std::vector<Eigen::MatrixXd>& sims) {
  // Compute the initial affinity matrices
  std::transform(
      Xs.begin(), Xs.end(), sims.begin(), [&](const Eigen::MatrixXd& X) {
        return affinityMat_(X);
      });

  // Compute the eigendecomposition of the Laplacian using only the
  // most informative view
  if (auto_num_clusters_) {
    Eigen::MatrixXd la_eigs_info_view(num_samples_, n_clusters_);
    Eigen::MatrixXd tmp_laplacian(num_samples_, num_samples_);
    double tmp_obj_val{};

    computeEigs_(sims[info_view_],   // X
                 n_clusters_,        // num_top_eigenvectors
                 la_eigs_info_view,  // u_mat
                 tmp_laplacian,      // laplacian
                 tmp_obj_val         // obj_val
    );

    // Apply the Zelnik-Manor and Perona (2004) method to compute the
    // number of clusters based on the data of the most informative
    // view
    ClusterRotate clusterrotate{};
    std::vector<std::vector<int>> clusters =
        clusterrotate.cluster(la_eigs_info_view);

    // Update the number of clusters to use in
    // subsequent clustering
    n_clusters_ = clusters.size();

  } else {
    // No determination of n_clusters_ (i.e. we use the provided
    // n_clusters as given)
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
  // Note that n_clusters_ might be updated here, depending on whether we have
  // set auto_num_clusters_ to true
  std::vector<Eigen::MatrixXd> sims(
      n_views_, Eigen::MatrixXd(num_samples_, num_samples_));
  fit_init_(Xs, sims);

  // Initialize matrices of eigenvectors U_v for each view
  // The matrix of top eigenvectors are of size num_samples_ x n_clusters_
  std::vector<Eigen::MatrixXd> U_mats(
      n_views_, Eigen::MatrixXd(num_samples_, n_clusters_));
  std::vector<Eigen::MatrixXd> tmp_laplacians(
      n_views_, Eigen::MatrixXd(num_samples_, num_samples_));
  std::vector<double> tmp_obj_vals(n_views_);
  std::vector<int> idx_views(n_views_);
  std::iota(idx_views.begin(), idx_views.end(), 0);

  std::for_each(std::execution::par_unseq,
                idx_views.begin(),
                idx_views.end(),
                [&](const int& view) {
                  return computeEigs_(sims[view],
                                      n_clusters_,
                                      U_mats[view],
                                      tmp_laplacians[view],
                                      tmp_obj_vals[view]);
                });

  // Iteratively compute new graph similarities, Laplacians and eigenvectors
  std::vector<Eigen::MatrixXd> eig_sums(
      n_views_, Eigen::MatrixXd(num_samples_, num_samples_));
  std::vector<Eigen::MatrixXd> new_sims(
      n_views_, Eigen::MatrixXd(num_samples_, num_samples_));
  std::vector<Eigen::MatrixXd> mat1(
      n_views_, Eigen::MatrixXd(num_samples_, num_samples_));

  int iter{0};
  while (iter < max_iter_) {
    // Compute the sums of the products of the spectral embeddings and their
    // transposes.
    // Note that each u_mat is of size num_samples x n_cluster. Hence,
    // each entry in eig_sums is num_samples x num_samples.
    std::transform(std::execution::par_unseq,
                   U_mats.begin(),
                   U_mats.end(),
                   eig_sums.begin(),
                   [&](const Eigen::MatrixXd& u_mat) {
                     return (u_mat * u_mat.transpose());
                   });

    Eigen::MatrixXd U_sum = Eigen::MatrixXd::Zero(num_samples_, num_samples_);

    std::for_each(eig_sums.begin(),
                  eig_sums.end(),
                  [&](const Eigen::MatrixXd& X) { return U_sum += X; });

    // Compute new graph similariity representation S_v
    std::iota(idx_views.begin(), idx_views.end(), 0);
    std::for_each(
        std::execution::par_unseq,
        idx_views.begin(),
        idx_views.end(),
        [&](const int& view) {
          Eigen::MatrixXd mat11 = sims[view] * (U_sum - eig_sums[view]);
          new_sims[view].noalias() = 0.5 * (mat11 + mat11.transpose());
        });

    // Recompute eigenvectors and get new U_v's
    std::for_each(std::execution::par_unseq,
                  idx_views.begin(),
                  idx_views.end(),
                  [&](const int& view) {
                    return computeEigs_(new_sims[view],
                                        n_clusters_,
                                        U_mats[view],
                                        tmp_laplacians[view],
                                        tmp_obj_vals[view]);
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
