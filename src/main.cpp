#include <Eigen/Dense>
#include <iostream>
#include <mlpack.hpp>

Eigen::MatrixXd cast_arma_to_eigen(arma::Mat<double> &arma_A) {
  Eigen::MatrixXd eigen_A = Eigen::Map<Eigen::MatrixXd>(
      arma_A.memptr(), arma_A.n_rows, arma_A.n_cols);

  return eigen_A;
}

arma::Mat<double> cast_eigen_to_arma(Eigen::MatrixXd &eigen_A) {
  auto arma_A = arma::Mat<double>(eigen_A.data(), eigen_A.rows(),
                                  eigen_A.cols(), false, false);

  return arma_A;
}

int main() {
  arma::Mat<double> data("0.539406815,0.843176636,0.472701471; \
                  0.212587646,0.351174901,0.81056695;  \
                  0.160147626,0.255047893,0.04072469;  \
                  0.564535197,0.943435462,0.597070812");

  Eigen::MatrixXd data_e{{0.539406815, 0.843176636, 0.472701471},
                         {0.212587646, 0.351174901, 0.81056695},
                         {0.160147626, 0.255047893, 0.04072469},
                         {0.564535197, 0.943435462, 0.597070812}};

  std::cout << data << std::endl;
  std::cout << data_e << std::endl;

  Eigen::MatrixXd data_a_to_e = cast_arma_to_eigen(data);
  arma::mat data_e_to_a = cast_eigen_to_arma(data_e);

  std::cout << "A to E"
            << "\n";
  std::cout << data_a_to_e << "\n";

  std::cout << "E to A \n"
            << "\n";
  std::cout << data_e_to_a << "\n";

  /* data = data.t(); */

  /* mlpack::NeighborSearch<mlpack::NearestNeighborSort,
   * mlpack::ManhattanDistance> */
  /*     nn(data); */
  /**/
  /* // Create the object we will store the nearest neighbors in. */
  /* arma::Mat<size_t> neighbors; */
  /* arma::mat distances; // We need to store the distance too. */
  /**/
  /* // Compute the neighbors. */
  /* nn.Search(1, neighbors, distances); */
  /**/
  /* // Write each neighbor and distance using Log. */
  /* for (size_t i = 0; i < neighbors.n_elem; ++i) { */
  /*   std::cout << "Nearest neighbor of point " << i << " is point " */
  /*             << neighbors[i] << " and the distance is " << distances[i] <<
   * "." */
  /*             << std::endl; */
  /* } */
  /**/
  return 0;
}
