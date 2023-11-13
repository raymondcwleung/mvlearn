#include <Eigen/Dense>
#include <cstddef>
#include <iostream>
#include <mlpack.hpp>

int main() {
  /* arma::Mat<double> data( */
  /*     "0.539406815,0.843176636,0.472701471; \ */
  /*     0.212587646,0.351174901,0.81056695;  \ */
  /*     0.160147626,0.255047893,0.04072469;  \ */
  /*     0.564535197,0.943435462,0.597070812"); */
  /* data = data.t(); */

  arma::Mat<double> data(
      "-1, -1; \
      -2, -1; \
      -3, -2; \
      1, 1; \
      2, 1; \
      3, 2");
  size_t num_obs = data.n_rows;
  size_t num_dim = data.n_cols;

  data = data.t();

  mlpack::NeighborSearch<mlpack::NearestNeighborSort, mlpack::EuclideanDistance>
      nn(data);

  arma::Mat<size_t> neighbors;
  arma::Mat<double> distances;

  int num_neighbors{3};
  nn.Search(num_neighbors, neighbors, distances);

  arma::Mat<size_t> wgt_edges(num_obs, num_obs, arma::fill::zeros);

  neighbors = neighbors.t();
  std::cout << neighbors << "\n" << std::endl;
  std::cout << distances.t() << "\n" << std::endl;

  for (size_t i = 0; i < num_obs; i++) {
    // The point itself is also connected to the
    // point itself
    wgt_edges(i, i) = 1;

    // We sum up to (num_neighbors - 1) and NOT
    // to num_neighbors --- this is because
    // we explicitly count in the point itself as a
    // neighbor.
    for (size_t j = 0; j < num_neighbors - 1; j++) {
      // Get the edge to the neighbors
      wgt_edges(i, neighbors(i, j)) = 1;
    }
  }

  std::cout << wgt_edges << "\n";

  /* int n_clusters{2}; */
  /* mvlearn::MVSpectralClustering mvsc {} */

  return 0;
}
