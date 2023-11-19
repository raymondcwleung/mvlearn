#ifndef SKLEARNCPP_METRICS_PAIRWISE_H_
#define SKLEARNCPP_METRICS_PAIRWISE_H_

#include <Eigen/Dense>

namespace sklearncpp::metrics::pairwise {

Eigen::MatrixXd rbfKernel(const Eigen::Ref<const Eigen::MatrixXd>& X,
                          const Eigen::Ref<const Eigen::MatrixXd>& Y,
                          double gamma);

}

#endif
