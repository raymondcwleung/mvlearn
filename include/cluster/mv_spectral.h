#ifndef MVLEARN_CLUSTER_MV_SPECTRAL_H
#define MVLEARN_CLUSTER_MV_SPECTRAL_H

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace mvlearn::cluster {

class MVSpectralClustering {
 private:
  double gamma_{};
  int n_neighbors_{};
  bool auto_num_clusters_{};

 protected:
  int n_clusters_{};
  int num_samples_{};
  int num_features_{};
  int info_view_{};
  std::string affinity_{};
  int max_iter_{};

  int n_views_{};

  void fit_init_(const std::vector<Eigen::MatrixXd>& Xs,
                 std::vector<Eigen::MatrixXd>& sims);

  Eigen::MatrixXd affinityMat_(const Eigen::Ref<const Eigen::MatrixXd>& X);

  Eigen::MatrixXd constructLaplacian_(
      const Eigen::Ref<const Eigen::MatrixXd>& X);

  void constructLaplacian_(const Eigen::Ref<const Eigen::MatrixXd>& X,
                           Eigen::MatrixXd& laplacian);

  Eigen::MatrixXd computeEigs_(const Eigen::Ref<const Eigen::MatrixXd>& X,
                               int num_top_eigenvectors);

  void computeEigs_(
      // inputs
      const Eigen::Ref<const Eigen::MatrixXd>& X,
      int num_top_eigenvectors,
      // outputs
      Eigen::MatrixXd& u_mat,
      Eigen::MatrixXd& laplacian,
      double& obj_val);

  /* Eigen::VectorXi fit_predict(const std::vector<Eigen::MatrixXd>& Xs); */

 public:
  Eigen::MatrixXd embedding_{};
  Eigen::VectorXi labels_{};

  MVSpectralClustering(int n_clusters,
                       int num_samples,
                       int num_features,
                       int info_view,
                       int max_iter,
                       std::string affinity,
                       int n_neighbors,
                       double gamma = -1,
                       bool auto_num_clusters = false);

  void fit(const std::vector<Eigen::MatrixXd>& Xs);
  Eigen::VectorXi fit_predict(const std::vector<Eigen::MatrixXd>& Xs);

  int get_num_clusters();
  /* void fit(const std::vector<EigenDRef>& Xs); */
};
}  // namespace mvlearn::cluster

#endif
