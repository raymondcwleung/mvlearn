#ifndef SCIPYCPP_SPATIAL_DISTANCE_H_
#define SCIPYCPP_SPATIAL_DISTANCE_H_

#include <Eigen/Dense>
/* #include <string> */

namespace scipycpp::spatial::distance {

Eigen::MatrixXd cdist(const Eigen::MatrixXd& XA, const Eigen::MatrixXd& XB);

}

#endif
