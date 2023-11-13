#ifndef SCIPYCPP_SPATIAL_DISTANCE_H_
#define SCIPYCPP_SPATIAL_DISTANCE_H_

#include <Eigen/Dense>
#include <string>

namespace scipycpp::spatial::distance {

template <typename Derived>
Eigen::MatrixXd cdist(const Eigen::MatrixBase<Derived>& XA,
                      const Eigen::MatrixBase<Derived>& XB,
                      const std::string metric = "euclidean");

}

#endif
