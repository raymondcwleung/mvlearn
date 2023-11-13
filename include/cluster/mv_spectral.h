#ifndef MVLEARN_CLUSTER_MV_SPECTRAL_H
#define MVLEARN_CLUSTER_MV_SPECTRAL_H

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace mvlearn::cluster {
class MVSpectralClustering {
 private:
  int n_clusters_{};
  int random_state_{};
  int info_view_{};
  int max_iter_{};
  int n_init_{};
  std::string affinity_{};
  float gamma_{};
  int n_neighbors_{};

  int n_samples_{};
  int n_features_{};

  Eigen::MatrixXd affinityMat_(const Eigen::Ref<const Eigen::MatrixXd>& X);
  Eigen::MatrixXd computeEigs_(const Eigen::Ref<const Eigen::MatrixXd>& X);

 public:
  MVSpectralClustering(int n_clusters, int random_state, int info_view,
                       int max_iter, int n_init, std::string affinity,
                       int n_neighbors, float gamma);

  void fit(const std::vector<Eigen::Ref<const Eigen::MatrixXd>>& Xs);
};
}  // namespace mvlearn::cluster

#endif
