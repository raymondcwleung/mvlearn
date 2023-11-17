#include "cluster/kmeans.h"

#include <Eigen/Dense>
#include <mlpack.hpp>

#include "utils_eigenarma/conversions.h"

namespace sklearn::cluster {

KMeans::KMeans(int n_clusters, int max_iter)
    : n_clusters_{n_clusters}, max_iter_{max_iter} {};

//! KMeans fit
/*!
 * \param X The data of the shape num_samples x num_features
 */
void KMeans::fit(const Eigen::Ref<const Eigen::MatrixXd>& X) {
  // Note the importance of transposing. Mlpack assumes we work with
  // a data matrix of the shape num_features x num_samples.
  arma::Mat<double> arma_X{
      utilseigenarma::castEigenToArma<double>(X.transpose())};

  // KMeans fit
  arma::Row<size_t> assignments;
  arma::Mat<double> centroids;
  mlpack::KMeans<> k;

  k.Cluster(arma_X, n_clusters_, assignments, centroids);

  assignments_ = assignments;
  centroids_ = centroids;
};

//! Given the new data, assign clusters
arma::Row<size_t> KMeans::assign(
    const Eigen::Ref<const Eigen::MatrixXd>& newX) {
  // Again, note the importance of transposing. Mlpack assumes we work with
  // a data matrix of the shape num_features x num_samples.
  arma::Mat<double> arma_newX{
      utilseigenarma::castEigenToArma<double>(newX.transpose())};

  arma::Row<size_t> new_assignments;
  new_assignments.set_size(arma_newX.n_cols);

  // Search
  mlpack::KNN a(centroids_);
  arma::Mat<double> resulting_distances;

  a.Search(arma_newX, 1, new_assignments, resulting_distances);

  return new_assignments;
};

}  // namespace sklearn::cluster
