#include <Eigen/Dense>
#include <cstddef>
#include <iostream>
#include <mlpack.hpp>

#include "Eigen/src/Core/DiagonalMatrix.h"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h"
#include "mlpack/core/metrics/lmetric.hpp"
#include "mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp"
#include "sklearncpp/neighbors/nearestneighbors.h"
#include "utils_eigenarma/conversions.h"

int main() {
  /* arma::Mat<double> data( */
  /*     "0.539406815,0.843176636,0.472701471; \ */
  /*     0.212587646,0.351174901,0.81056695;  \ */
  /*     0.160147626,0.255047893,0.04072469;  \ */
  /*     0.564535197,0.943435462,0.597070812"); */
  /* data = data.t(); */

  Eigen::MatrixXd data_eigen{{-1.0, -1.0}, {-2.0, -1}, {-3, -2},
                             {1, 1},       {2, 1},     {3, 2}};

  /* std::cout << data_eigen << std::endl; */

  int n_clusters_{5};

  /* double lo{0}; */
  /* double hi{1}; */
  /* double range{hi - lo}; */
  /* std::srand(12345); */
  /* Eigen::MatrixXd X = Eigen::MatrixXd::Random(10, 10); */
  /* X = (X + Eigen::MatrixXd::Constant(10, 10, 1.)) * range / 2; */
  /* X = (X + Eigen::MatrixXd::Constant(10, 10, lo)); */

  Eigen::MatrixXd X{{0.178395, 0.367737, 0.655419, 0.41206, 0.576451, 0.485838,
                     0.879841, 0.182721, 0.578371, 0.554799},
                    {0.399677, 0.222367, 0.527644, 0.81593, 0.757953, 0.0252066,
                     0.0174133, 0.472538, 0.703867, 0.793983},
                    {0.166599, 0.219657, 0.505857, 0.280523, 0.176208, 0.69734,
                     0.564316, 0.650993, 0.974248, 0.783903},
                    {0.212122, 0.201738, 0.915692, 0.578658, 0.796108, 0.991695,
                     0.160365, 0.35893, 0.275711, 0.119115},
                    {0.0619357, 0.0441157, 0.188085, 0.0280522, 0.959691,
                     0.940898, 0.596072, 0.268646, 0.695562, 0.954347},
                    {0.0541498, 0.34392, 0.119156, 0.342459, 0.220324, 0.885425,
                     0.592369, 0.610684, 0.915147, 0.379974},
                    {0.275665, 0.647654, 0.0580968, 0.632808, 0.140028,
                     0.110851, 0.502824, 0.579253, 0.161136, 0.711484},
                    {0.0477572, 0.149464, 0.719929, 0.303718, 0.607345,
                     0.998995, 0.22888, 0.408675, 0.806413, 0.457171},
                    {0.321033, 0.0296677, 0.637535, 0.390216, 0.369788,
                     0.605354, 0.896086, 0.218028, 0.914142, 0.608854},
                    {0.272734, 0.878494, 0.880846, 0.953841, 0.169696, 0.748386,
                     0.89304, 0.949041, 0.766489, 0.60757}};
  X = X.transpose() * X;

  // Compute the normalized Laplacian
  Eigen::VectorXd col_sums = X.colwise().sum();
  Eigen::MatrixXd d_mat = Eigen::MatrixXd(X.colwise().sum().asDiagonal());
  Eigen::MatrixXd d_alt = d_mat.inverse().cwiseSqrt();
  Eigen::MatrixXd laplacian = d_alt * X * d_alt;

  // Make the resulting matrix symmetric
  laplacian = (laplacian + laplacian.transpose()) / 2.0;

  // To ensure PSD
  double min_val = laplacian.minCoeff();
  if (min_val < 0.0) {
    laplacian = laplacian + Eigen::MatrixXd::Constant(
                                laplacian.rows(), laplacian.cols(), min_val);
  }

  // Obtain the top n_cluster eigenvectors the of the Laplacian
  // Note Eigen::SelfAdjointEigenSolver sorts the eigenvalues in increasing
  // order
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(laplacian);
  Eigen::MatrixXd u_mat{es.eigenvectors()};

  Eigen::MatrixXd la_eigs =
      u_mat(Eigen::all, Eigen::seq(u_mat.cols() - n_clusters_, Eigen::last));

  /* std::cout << es.eigenvalues() << "\n"; */
  /* std::cout << la_eigs << std::endl; */

  arma::Mat<double> arma_X{utilseigenarma::castEigenToArma(X)};

  int num_clusters{3};
  arma::Row<size_t> assignments;
  arma::Mat<double> centroids;

  mlpack::KMeans<> k;
  k.Cluster(arma_X, num_clusters, assignments, centroids);

  std::cout << centroids << "\n";

  Eigen::MatrixXd newX0 = Eigen::MatrixXd::Random(7, 10);
  Eigen::MatrixXd newX = newX0.transpose();  // 10 x 7
  // i.e. "data" in mlpack must be of the form space_dim x num_samples
  // (this is the TRANSPOSE of the usual design matrix form)

  newX = X(Eigen::all, Eigen::seq(2, 5));

  arma::Mat<double> arma_newX{utilseigenarma::castEigenToArma(newX)};

  arma::Row<size_t> new_assignments;
  new_assignments.set_size(arma_newX.n_cols);

  mlpack::KNN a(centroids);
  arma::Mat<double> resulting_distances;

  a.Search(arma_newX, 1, new_assignments, resulting_distances);

  std::cout << new_assignments << "\n";
  std::cout << resulting_distances << "\n";

  /* arma::Mat<double> data( */
  /*     "-1, -1; \ */
  /*     -2, -1; \ */
  /*     -3, -2; \ */
  /*     1, 1; \ */
  /*     2, 1; \ */
  /*     3, 2"); */
  /* size_t num_obs = data.n_rows; */
  /* size_t num_dim = data.n_cols; */
  /**/
  /* data = data.t(); */
  /**/
  /* mlpack::NeighborSearch<mlpack::NearestNeighborSort,
   * mlpack::EuclideanDistance> */
  /*     nn(data); */
  /**/
  /* arma::Mat<size_t> neighbors; */
  /* arma::Mat<double> distances; */
  /**/
  /* int num_neighbors{3}; */
  /* nn.Search(num_neighbors, neighbors, distances); */
  /**/
  /* arma::Mat<size_t> wgt_edges(num_obs, num_obs, arma::fill::zeros); */
  /**/
  /* neighbors = neighbors.t(); */
  /* std::cout << neighbors << "\n" << std::endl; */
  /* std::cout << distances.t() << "\n" << std::endl; */
  /**/
  /* for (size_t i = 0; i < num_obs; i++) { */
  /*   // The point itself is also connected to the */
  /*   // point itself */
  /*   wgt_edges(i, i) = 1; */
  /**/
  /*   // We sum up to (num_neighbors - 1) and NOT */
  /*   // to num_neighbors --- this is because */
  /*   // we explicitly count in the point itself as a */
  /*   // neighbor. */
  /*   for (size_t j = 0; j < num_neighbors - 1; j++) { */
  /*     // Get the edge to the neighbors */
  /*     wgt_edges(i, neighbors(i, j)) = 1; */
  /*   } */
  /* } */
  /**/
  /* std::cout << wgt_edges << "\n"; */

  /* int n_clusters{2}; */
  /* mvlearn::MVSpectralClustering mvsc {} */

  return 0;
}
