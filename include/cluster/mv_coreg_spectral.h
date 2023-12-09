#ifndef MVLEARN_CLUSTER_MV_COREG_SPECTRAL_H
#define MVLEARN_CLUSTER_MV_COREG_SPECTRAL_H

#include <Eigen/Dense>

#include "cluster/mv_spectral.h"

namespace mvlearn::cluster {

class MVCoRegSpectralClustering
    : public mvlearn::cluster::MVSpectralClustering {
 private:
  double v_lambda_{};

 public:
  Eigen::MatrixXd embedding_{};
  Eigen::VectorXi labels_{};

  MVCoRegSpectralClustering(int n_clusters, int num_samples, int num_features,
                            int info_view, int max_iter, std::string affinity,
                            int n_neighbors, double gamma = -1,
                            bool auto_num_clusters = false,
                            double v_lambda = 0.5);

  void fit(const std::vector<Eigen::MatrixXd>& Xs);
  Eigen::VectorXi fit_predict(const std::vector<Eigen::MatrixXd>& Xs);
};

}  // namespace mvlearn::cluster

#endif
