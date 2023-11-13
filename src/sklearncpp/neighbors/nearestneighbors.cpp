#include "neighbors/nearestneighbors.h"

#include <mlpack.hpp>

namespace sklearncpp::neighbors {

template <typename SortPolicy, typename MetricType>
Eigen::MatrixXi nearestNeighbors(const Eigen::Ref<const Eigen::MatrixXd>& data,
                                 const SortPolicy& sortpolicy,
                                 const MetricType& metrictype,
                                 int num_neighbors) {
  // Take the Eigen based matrix data and convert it to an Armadillo matrix
}

}  // namespace sklearncpp::neighbors
