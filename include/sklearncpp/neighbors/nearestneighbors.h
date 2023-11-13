#ifndef SKLEARNCPP_NEIGHBORS_NEARESTNEIGHBORS_H_
#define SKLEARNCPP_NEIGHBORS_NEARESTNEIGHBORS_H_

#include <Eigen/Dense>
#include <mlpack.hpp>

namespace sklearncpp::neighbors {
template <typename SortPolicy, typename MetricType>
Eigen::MatrixXi nearestNeighbors(const Eigen::Ref<const Eigen::MatrixXd>& data,
                                 const SortPolicy& sortpolicy,
                                 const MetricType& metrictype,
                                 int num_neighbors);

}

#endif
