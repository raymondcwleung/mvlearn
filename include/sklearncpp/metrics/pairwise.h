#ifndef SKLEARNCPP_METRICS_PAIRWISE_H_
#define SKLEARNCPP_METRICS_PAIRWISE_H_

#include <Eigen/Dense>

namespace sklearncpp::metrics::pairwise {

Eigen::MatrixXd rbfKernel(const Eigen::Ref<const Eigen::MatrixXd>& X,
                          const Eigen::Ref<const Eigen::MatrixXd>& Y,
                          double gamma);

Eigen::MatrixXd rbfKernel(const Eigen::Ref<const Eigen::MatrixXd>& X,
                          double gamma);

Eigen::MatrixXd rbfLocalKernel(const Eigen::Ref<const Eigen::MatrixXd>& X,
                               int num_neighbors = 7);

}  // namespace sklearncpp::metrics::pairwise

#endif
