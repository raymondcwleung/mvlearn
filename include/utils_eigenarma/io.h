#ifndef UTILS_EIGENARMA_IO_
#define UTILS_EIGENARMA_IO_

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <mlpack/core.hpp>
#include <string>

namespace utilseigenarma {

//! Save a Eigen matrix to CSV
/*! https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
 *
 * \param file_name Filename
 * \param matrix A Eigen double matrix
 */
template <typename T>
void saveData(std::string file_name,
              Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix) {
  const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
                                         Eigen::DontAlignCols, ", ", "\n");
  std::ofstream file(file_name);
  if (file.is_open()) {
    file << matrix.format(CSVFormat);
    file.close();
  }
}

//! Load a CSV to an Eigen matrix
template <typename T>
Eigen::MatrixXd loadData(const std::string& path) {
  arma::mat X;
  X.load(path, arma::csv_ascii);

  return Eigen::Map<const T>(X.memptr(), X.n_rows, X.n_cols);
}

}  // namespace utilseigenarma
//
#endif  // !UTILS_EIGENARMA_IO_
