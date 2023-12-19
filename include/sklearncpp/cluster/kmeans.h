#ifndef SKLEARNCPP_CLUSTER_KMEANS_H_
#define SKLEARNCPP_CLUSTER_KMEANS_H_

#include <Eigen/Dense>
#include <armadillo>

namespace sklearn::cluster {

class KMeans {
 public:
  KMeans(int n_clusters, int max_iter, size_t seed = 123456);

  Eigen::VectorXi fit_predict(const Eigen::Ref<const Eigen::MatrixXd>& X);

 private:
  int n_clusters_{};
  int max_iter_{};
  int num_samples_{};
  int num_features_{};
  size_t seed_{};

  arma::Mat<double> centroids_{};
  arma::Row<size_t> assignments_{};
  arma::Mat<double> arma_X_transposed_{};

  arma::Row<size_t> assign(const arma::Mat<double>& newX);
  void fit(const Eigen::Ref<const Eigen::MatrixXd>& X);
};

}  // namespace sklearn::cluster

#endif
