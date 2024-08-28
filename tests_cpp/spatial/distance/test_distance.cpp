#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include "spatial/distance/distance.h"
#include "utils/utils_eigen.cpp"

TEST_CASE("Euclidean distance between a set of vectors", "[cdist]") {
  Eigen::MatrixXd coords{{35.0456, -85.2672},
                         {35.1174, -89.9711},
                         {35.9728, -83.9422},
                         {36.1667, -86.7833}};

  Eigen::MatrixXd calc = spatial::distance::cdist(coords, coords);

  Eigen::MatrixXd ans =
      utils::utilseigen::loadData<Eigen::MatrixXd>("./test_cdist.txt");

  REQUIRE(calc.isApprox(ans) == true);
}
