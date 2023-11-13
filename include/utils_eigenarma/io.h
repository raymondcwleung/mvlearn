#ifndef UTILS_EIGENARMA_IO_
#define UTILS_EIGENARMA_IO_

#include <Eigen/Dense>
#include <string>

namespace utilseigenarma {

void saveData(std::string file_name, Eigen::MatrixXd matrix);

template <typename T>
Eigen::MatrixXd loadData(const std::string& path);

}  // namespace utilseigenarma

#endif  // !UTILS_EIGENARMA_IO_
