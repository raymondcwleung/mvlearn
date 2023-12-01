
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <cstddef>
#include <iostream>
#include <map>
#include <random>

#include "cluster/mv_spectral.h"
#include "sklearncpp/cluster/kmeans.h"
#include "sklearncpp/neighbors/nearestneighbors.h"
#include "utils_eigenarma/conversions.h"

int main() {
  /* arma::Mat<double> data( */
  /*     "0.539406815,0.843176636,0.472701471; \ */
  /*     0.212587646,0.351174901,0.81056695;  \ */
  /*     0.160147626,0.255047893,0.04072469;  \ */
  /*     0.564535197,0.943435462,0.597070812"); */
  /* data = data.t(); */

  std::random_device dev;
  std::mt19937 rng(dev());

  /* std::cout << data_eigen << std::endl; */

  int num_samples{10};

  double lo{0};
  double hi{1};
  double range{hi - lo};
  std::srand(12345);
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(num_samples, 6).array().cos();

  Eigen::MatrixXd X0 = X;
  Eigen::MatrixXd X1 = X0.array().sin();

  /* arma::Mat<double> A(50, 7, arma::fill::randu); */

  int n_cluster{3};
  int num_features{6};
  int random_state{-1};  // Fix
  int info_view{0};
  int max_iter{1};
  int n_init{-1};  // Fix
  std::string affinity{"rbf_local_scale"};
  /* std::string affinity{"rbf"}; */
  int n_neighbors{2};
  double gamma{-1};

  std::vector<Eigen::MatrixXd> Xs{X0, X1};

  mvlearn::cluster::MVSpectralClustering mvsc(
      n_cluster, num_samples, num_features, random_state, info_view, max_iter,
      n_init, affinity, n_neighbors, gamma);

  mvsc.fit(Xs);

  return 0;
}
