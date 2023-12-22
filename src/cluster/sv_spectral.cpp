#include "cluster/sv_spectral.h"

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
#include "cluster/mv_spectral.h"
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

SVSpectralClustering::SVSpectralClustering(int n_clusters,
                                           int num_samples,
                                           int num_features,
                                           int max_iter,
                                           std::string affinity,
                                           int n_neighbors,
                                           double gamma,
                                           bool auto_num_clusters,
                                           bool use_spectra)
    : mvlearn::cluster::MVSpectralClustering(n_clusters,
                                             num_samples,
                                             num_features,
                                             -1,  // info_view
                                             max_iter,
                                             affinity,
                                             n_neighbors,
                                             gamma,
                                             auto_num_clusters,
                                             use_spectra){};

//! Setup the initial affinity matrices at the beginning of the
//! fitting procedure
void SVSpectralClustering::fit_init_(const Eigen::MatrixXd& X,
                                     Eigen::MatrixXd& sim) {
  // Compute the initial affinity matrices
  sim = affinityMat_(X);

  // Compute the eigendecomposition of the Laplacian using only the
  // most informative view
  if (auto_num_clusters_) {
    Eigen::MatrixXd la_eigs(num_samples_, n_clusters_);
    Eigen::MatrixXd tmp_laplacian(num_samples_, num_samples_);
    double tmp_obj_val{};

    computeEigs_(sim,            // X
                 n_clusters_,    // num_top_eigenvectors
                 la_eigs,        // u_mat
                 tmp_laplacian,  // laplacian
                 tmp_obj_val     // obj_val
    );

    // Apply the Zelnik-Manor and Perona (2004) method to compute the
    // number of clusters based on the data of the most informative
    // view
    ClusterRotate clusterrotate{};
    std::vector<std::vector<int>> clusters = clusterrotate.cluster(la_eigs);

    // Update the number of clusters to use in
    // subsequent clustering
    n_clusters_ = clusters.size();

  } else {
    // No determination of n_clusters_ (i.e. we use the provided
    // n_clusters as given)
  }
};

void SVSpectralClustering::fit(const Eigen::MatrixXd& X) {
  // Compute the similarity matrix W
  Eigen::MatrixXd sim(num_samples_, num_samples_);
  fit_init_(X, sim);

  // Initialize matrices of eigenvectors U
  Eigen::MatrixXd U_mat(num_samples_, n_clusters_);
  Eigen::MatrixXd tmp_laplacian(num_samples_, num_samples_);
  double tmp_obj_val{};
  computeEigs_(sim,            // X
               n_clusters_,    // num_top_eigenvectors
               U_mat,          // u_mat
               tmp_laplacian,  // laplacian
               tmp_obj_val     // obj_val
  );

  // Iteratively compute new graph similarities, Laplacians and eigenvectors
  Eigen::MatrixXd eig_sum(num_samples_, num_samples_);
  Eigen::MatrixXd new_sim(num_samples_, num_samples_);

  int iter{0};
  while (iter < max_iter_) {
    // Compute the sums of the products of the spectral embeddings and their
    // transposes.
    // Note that each u_mat is of size num_samples x n_cluster. Hence,
    // each entry in eig_sums is num_samples x num_samples.
    eig_sum.noalias() = U_mat * U_mat.transpose();

    // Compute new graph similariity representation S_v
    Eigen::MatrixXd mat11 = sim * eig_sum;
    new_sim = 0.5 * (mat11 + mat11.transpose());

    // Recompute eigenvectors and get new U
    computeEigs_(new_sim,        // X
                 n_clusters_,    // num_top_eigenvectors
                 U_mat,          // u_mat
                 tmp_laplacian,  // laplacian
                 tmp_obj_val     // obj_val
    );

    iter++;
  }

  // Row normalize
  for (int j = 0; j < U_mat.rows(); j++) {
    U_mat.row(j).normalize();
  }

  // Perform k-means clustering
  sklearn::cluster::KMeans kmeans(n_clusters_, max_iter_);
  embedding_ = U_mat;
  labels_ = kmeans.fit_predict(embedding_);
}

Eigen::VectorXi mvlearn::cluster::SVSpectralClustering::fit_predict(
    const Eigen::MatrixXd& X) {
  fit(X);

  return labels_;
}

}  // namespace mvlearn::cluster
