#include "cluster/mv_spectral.h"

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <mlpack.hpp>
#include <string>

/* #include "metrics/pairwise/pairwise.h" */

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

    // Compute the median of the distances matrix
    arma::Mat<double> arma_X = utilseigenarma::castEigenToArma<double>(X);
    arma::Col<double> arma_vecX = arma::vectorise(arma_X);
    double median = arma::median(arma_vecX);

    gamma = 1.0 / (2.0 * std::pow(median, 2));
  } else {
    gamma = gamma_;
  }

  // Produce the affinity matrix based on the selected kernel type
  /* Eigen::MatrixXd sims(num_samples_, num_samples_); */
  Eigen::MatrixXd sims;

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

  Eigen::MatrixXd la_eigs(num_samples_, n_clusters_);
  la_eigs =
      u_mat(Eigen::all, Eigen::seq(u_mat.cols() - n_clusters_, Eigen::last));

  return la_eigs;
}

Eigen::MatrixXd rbfKernel(const Eigen::Ref<const Eigen::MatrixXd>& X,
                          const Eigen::Ref<const Eigen::MatrixXd>& Y,
                          double gamma) {
  // NOTE:: We're not checking that X and Y have the same shapes
  Eigen::MatrixXd Kmat(X.rows(), X.rows());

  for (int i; i < X.rows(); i++) {
    for (int j; j < Y.rows(); j++) {
      Kmat(i, j) = std::exp(
          -1.0 * gamma * (X(i, Eigen::all) - Y(j, Eigen::all)).squaredNorm());
    }
  }

  return Kmat;
}

//! Performs clustering on the multiple views of data
/*
 *
 */
void MVSpectralClustering::fit(const Eigen::Ref<const Eigen::MatrixXd>& X0,
                               const std::map<int, Eigen::MatrixXd>& Xs) {
  n_views_ = Xs.size();

  // Compute the similarity matrices
  std::map<int, Eigen::MatrixXd> sims;

  Eigen::MatrixXd blah = this->rbfKernel(X0, X0);
  std::cout << blah << "\n";

  /* for (auto const& [view, X] : Xs) { */
  /*   sims[view] = affinityMat_(X); */
  /* } */

  /* std::transform(Xs.begin(), Xs.end(), sims.begin(), */
  /*                [this](auto X) { return affinityMat_(X); }); */

  /**/
  /* // Initialize matrices of eigenvectors */
  /* std::vector<Eigen::MatrixXd> U_mats(n_views_); */
  /* std::transform(sims.begin(), sims.end(), U_mats.begin(), */
  /*                [this](const Eigen::MatrixXd& X) { return computeEigs_(X);
   * }); */
  /**/
  /* // Iteratively compute new graph similarities, Laplacians and eigenvectors
   */
  /* int iter{0}; */
  /* std::vector<Eigen::MatrixXd> eig_sums(n_views_); */
  /* Eigen::MatrixXd U_sum{Eigen::MatrixXd::Zero(num_samples_, n_clusters_)}; */
  /* while (iter < max_iter_) { */
  /*   // Compute the sums of the products of the spectral embeddings and their
   */
  /*   // transposes */
  /*   std::transform(U_mats.begin(), U_mats.end(), eig_sums.begin(), */
  /*                  [this](const Eigen::MatrixXd& u_mat) { */
  /*                    return u_mat * u_mat.transpose(); */
  /*                  }); */
  /*   for (auto& u_mat : U_mats) { */
  /*     U_sum += u_mat; */
  /*   } */
  /**/
  /*   std::vector<Eigen::MatrixXd> new_sims{}; */
  /*   Eigen::MatrixXd mat1{}; */
  /**/
  /*   for (int view = 0; view < n_views_; view++) { */
  /*     // Compute new graph similariity representation */
  /*     mat1 = sims[view] * (U_sum - eig_sums[view]); */
  /*     mat1 = (mat1 + mat1.transpose()) / 2.0; */
  /**/
  /*     new_sims.push_back(mat1); */
  /**/
  /*     // Recompute eigenvectors */
  /*     std::transform(new_sims.begin(), new_sims.end(), U_mats.begin(), */
  /*                    [this](const Eigen::Ref<const Eigen::MatrixXd>& X) { */
  /*                      return computeEigs_(X); */
  /*                    }); */
  /*   } */
  /**/
  /*   iter++; */
  /* } */
  /**/
  /* // Row normalize */
  /* Eigen::VectorXd U_norm(n_clusters_); */
  /* for (int view = 0; view < n_views_; view++) { */
  /*   for (int j = 0; j < U_mats[view].cols(); j++) { */
  /*     U_mats[view].col(j).normalize(); */
  /*   } */
  /* } */

  // Perform k-means clustering
  /* sklearn::cluster::KMeans kmeans(n_clusters_, 1); */
  /* embedding_ = U_mats[info_view_]; */
  /* kmeans.fit(embedding_); */
  /* arma::Row<size_t> arma_labels{kmeans.assign(embedding_)}; */
}

}  // namespace mvlearn::cluster
