#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include "metrics/pairwise/pairwise.h"
#include "utils/utils_eigen.cpp"

TEST_CASE("RBF KERNEL") {
  Eigen::MatrixXd coords{{35.0456, -85.2672},
                         {35.1174, -89.9711},
                         {35.9728, -83.9422},
                         {36.1667, -86.7833}};
  double gamma{1.0};

  Eigen::MatrixXd calc = metrics::pairwise::rbfKernel(coords, coords, gamma);

  Eigen::MatrixXd ans =
      utils::utilseigen::loadData<Eigen::MatrixXd>("./test_rbfKernel.txt");

  REQUIRE(calc.isApprox(ans) == true);
}
