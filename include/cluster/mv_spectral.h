#ifndef MVLEARN_CLUSTER_MV_SPECTRAL_H
#define MVLEARN_CLUSTER_MV_SPECTRAL_H

#include <Eigen/Dense>
#include <armadillo>
#include <map>
#include <string>

namespace mvlearn::cluster {
class MVSpectralClustering {
 private:
  int n_clusters_{};
  int num_samples_{};
  int num_features_{};
  int random_state_{};
  int info_view_{};
  int max_iter_{};
  int n_init_{};
  std::string affinity_{};
  double gamma_{};
  int n_neighbors_{};

  int n_views_{};

  Eigen::MatrixXd affinityMat_(const Eigen::MatrixXd& X);
  Eigen::MatrixXd computeEigs_(const Eigen::Ref<const Eigen::MatrixXd>& X);

 public:
  Eigen::MatrixXd embedding_{};
  Eigen::VectorXi labels_{};

  MVSpectralClustering(int n_clusters, int num_samples, int num_features,
                       int random_state, int info_view, int max_iter,
                       int n_init, std::string affinity, int n_neighbors,
                       double gamma = -1);

  void fit(const std::vector<Eigen::MatrixXd>& Xs);
};
}  // namespace mvlearn::cluster

#endif
