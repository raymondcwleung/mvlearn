#ifndef MVLEARN_CLUSTER_SV_SPECTRAL_H
#define MVLEARN_CLUSTER_SV_SPECTRAL_H

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "cluster/mv_spectral.h"

namespace mvlearn::cluster {

class SVSpectralClustering : public mvlearn::cluster::MVSpectralClustering {
 public:
  Eigen::MatrixXd embedding_{};
  Eigen::VectorXi labels_{};

  SVSpectralClustering(int n_clusters,
                       int num_samples,
                       int num_features,
                       int max_iter,
                       std::string affinity,
                       int n_neighbors,
                       double gamma = -1,
                       bool auto_num_clusters = false,
                       bool use_spectra = true);

  void fit(const Eigen::MatrixXd& X);
  void fit_init_(const Eigen::MatrixXd& X, Eigen::MatrixXd& sim);
  Eigen::VectorXi fit_predict(const Eigen::MatrixXd& X);
};

}  // namespace mvlearn::cluster

#endif
