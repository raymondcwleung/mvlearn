#include "cluster/kmeans.h"

#include <assert.h>

#include <Eigen/Dense>
#include <mlpack.hpp>

#include "Eigen/src/Core/Matrix.h"
#include "utils_eigenarma/conversions.h"

namespace sklearn::cluster {

KMeans::KMeans(int n_clusters, int max_iter, size_t seed)
    : n_clusters_{n_clusters}, max_iter_{max_iter}, seed_{seed} {};

//! KMeans fit
/*!
 * \param X The data of the shape num_samples x num_features
 */
void KMeans::fit(const Eigen::Ref<const Eigen::MatrixXd>& X) {
  // Convert from Eigen to Armadillo matrix.
  // Note that it is critical to transpose the input data.
  // The input X (Eigen::MatrixXd) is of dimensions num_samples x num_features.
  // However, the mlpack::KMeans methods assume the data matrices are of the
  // shape num_features x num_samples.

  /* std::cout << "X rows" << X.rows() << "\n"; */
  /* std::cout << "X cols" << X.cols() << "\n"; */

  num_samples_ = X.rows();
  num_features_ = X.cols();
  arma_X_transposed_ = utilseigenarma::castEigenToArma<double>(X).t();

  /* std::cout << "num_samples_" << num_samples_ << "\n"; */
  /* std::cout << "num_features_" << num_samples_ << "\n"; */

  mlpack::RandomSeed(seed_);

  /* std::cout << "Let's go Kmeans fit" */
  /*           << "\n"; */
  /**/
  /* std::cout << "arma_X_transposed_ rows" << arma_X_transposed_.n_rows <<
   * "\n"; */
  /* std::cout << "arma_X_transposed_ cols" << arma_X_transposed_.n_cols <<
   * "\n"; */

  // KMeans fit
  mlpack::KMeans<> km;

  /* std::cout << "Am I OK?" */
  /*           << "\n"; */

  km.Cluster(arma_X_transposed_,  // data
             n_clusters_,         // clusters
             assignments_,        // assignments
             centroids_           // centroids
  );

  /* std::cout << "Done Kmeans fit" */
  /*           << "\n"; */
};

//! Given the new data, assign clusters
arma::Row<size_t> KMeans::assign(const arma::Mat<double>& newX) {
  // The number of clusters must equal to the
  // number of samples of the input matrix X
  assert(newX.n_cols == num_samples_);

  // Search
  arma::Row<size_t> new_assignments;
  arma::Mat<double> resulting_distances{};
  new_assignments.set_size(newX.n_cols);
  mlpack::KNN knn(centroids_);
  knn.Search(newX, 1, new_assignments, resulting_distances);

  return new_assignments;
};

Eigen::VectorXi KMeans::fit_predict(
    const Eigen::Ref<const Eigen::MatrixXd>& X) {
  // First fit
  this->fit(X);

  /* std::cout << "All done Kmeans fit" */
  /*           << "\n"; */

  // Now assign clusters
  arma::Row<size_t> new_assignments{this->assign(arma_X_transposed_)};

  // Return as Eigen::VectorXi
  Eigen::VectorXi e_new_assignments(num_samples_);
  for (int i = 0; i < new_assignments.size(); i++) {
    e_new_assignments[i] = new_assignments(i);
  }

  return e_new_assignments;
};

}  // namespace sklearn::cluster
