#include "neighbors/nearestneighbors.h"

#include <cstddef>
#include <iostream>
#include <mlpack.hpp>

#include "utils_eigenarma/conversions.h"

namespace sklearncpp::neighbors {

template <class SortPolicy, class MetricType>
Eigen::MatrixXi nearestNeighbors(const Eigen::Ref<const Eigen::MatrixXd>& data,
                                 int num_neighbors, int num_obs) {
  // Take the Eigen based matrix data and convert it to an Armadillo matrix
  arma::Mat<double> arma_data = utilseigenarma::castEigenToArma<double>(data);

  // Critical to take the transpose for mlpack::NeighborSearch
  arma_data = arma_data.t();

  // Prep for mlpack
  mlpack::NeighborSearch<SortPolicy, MetricType> nn(arma_data);
  arma::Mat<std::size_t> neighbors;
  arma::Mat<double> distances;
  nn.Search(num_neighbors, neighbors, distances);

  // Construct the edges
  arma::Mat<std::size_t> wgt_edges(num_obs, num_obs, arma::fill::zeros);

  // Must transpose back
  neighbors = neighbors.t();

  // Fill in the connectivity matrix
  for (std::size_t i = 0; i < num_obs; i++) {
    // The point itself is also connected to the
    // point itself
    wgt_edges(i, i) = 1;

    // We sum up to (num_neighbors - 1) and NOT
    // to num_neighbors --- this is because
    // we explicitly count in the point itself as a
    // neighbor.
    for (std::size_t j = 0; j < num_neighbors - 1; j++) {
      // Get the edge to the neighbors
      wgt_edges(i, neighbors(i, j)) = 1;
    }
  }

  // Convert back to Eigen
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> wgt_edges_eigen =
      utilseigenarma::castArmaToEigen<int>(
          arma::conv_to<arma::Mat<int>>::from(wgt_edges));

  return wgt_edges_eigen;
}

}  // namespace sklearncpp::neighbors

template Eigen::MatrixXi sklearncpp::neighbors::nearestNeighbors<
    mlpack::NearestNeighborSort, mlpack::EuclideanDistance>(
    const Eigen::Ref<const Eigen::MatrixXd>&, int, int);

template Eigen::MatrixXi sklearncpp::neighbors::nearestNeighbors<
    mlpack::NearestNeighborSort, mlpack::ManhattanDistance>(
    const Eigen::Ref<const Eigen::MatrixXd>&, int, int);
