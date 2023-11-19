#ifndef SKLEARNCPP_CLUSTER_KMEANS_H_
#define SKLEARNCPP_CLUSTER_KMEANS_H_

#include <Eigen/Dense>
#include <mlpack.hpp>

namespace sklearn::cluster {

class KMeans {
 public:
  KMeans(int n_clusters, int max_iter);
  void fit(const Eigen::Ref<const Eigen::MatrixXd>& X);
  arma::Row<size_t> assign(const arma::Mat<double>& newX);
  Eigen::VectorXi assign(const Eigen::Ref<const Eigen::MatrixXd>& newX);

 private:
  int n_clusters_{};
  int max_iter_{};
  arma::Row<size_t> assignments_{};
  arma::Mat<double> centroids_{};
};

}  // namespace sklearn::cluster

#endif
