#ifndef SKLEARNCPP_NEIGHBORS_NEARESTNEIGHBORS_H_
#define SKLEARNCPP_NEIGHBORS_NEARESTNEIGHBORS_H_

#include <Eigen/Dense>
#include <mlpack.hpp>

namespace sklearncpp::neighbors {
template <class SortPolicy, class MetricType>
Eigen::MatrixXi nearestNeighbors(const Eigen::Ref<const Eigen::MatrixXd>& data,
                                 int num_neighbors);

}  // namespace sklearncpp::neighbors

#endif
