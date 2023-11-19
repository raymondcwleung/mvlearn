#include "cluster/kmeans.h"

#include <Eigen/Dense>
#include <mlpack.hpp>

#include "Eigen/src/Core/Matrix.h"
#include "utils_eigenarma/conversions.h"

namespace sklearn::cluster {

KMeans::KMeans(int n_clusters, int max_iter)
    : n_clusters_{n_clusters}, max_iter_{max_iter} {};

//! KMeans fit
/*!
 * \param X The data of the shape num_samples x num_features
 */
void KMeans::fit(const Eigen::Ref<const Eigen::MatrixXd>& X) {
  // Convert from Eigen to Armadillo matrix
  arma::Mat<double> arma_X{utilseigenarma::castEigenToArma<double>(X)};

  // KMeans fit
  mlpack::KMeans<> km;
  km.Cluster(arma_X, n_clusters_, assignments_, centroids_);
};

//! Given the new data, assign clusters
arma::Row<size_t> KMeans::assign(const arma::Mat<double>& newX) {
  // Search
  arma::Row<size_t> new_assignments;
  arma::Mat<double> resulting_distances{};
  new_assignments.set_size(newX.n_cols);

  mlpack::KNN knn(centroids_);
  knn.Search(newX, 1, new_assignments, resulting_distances);

  return new_assignments;
};

Eigen::VectorXi KMeans::assign(const Eigen::Ref<const Eigen::MatrixXd>& newX) {
  // Convert from Eigen to Armadillo matrix
  arma::Mat<double> arma_newX{utilseigenarma::castEigenToArma<double>(newX)};

  arma::Row<size_t> arma_new_assignments{KMeans::assign(arma_newX)};

  arma::Row<int> blah = arma::conv_to<int>::from(arma_new_assignments);

  return utilseigenarma::castArmaToEigen<int>();
};

}  // namespace sklearn::cluster
