#include "cluster/mv_coreg_spectral.h"

#include <Eigen/src/Core/Matrix.h>

#include <Eigen/Dense>
#include <execution>
#include <iostream>

#include "sklearncpp/cluster/kmeans.h"

namespace mvlearn::cluster {

MVCoRegSpectralClustering::MVCoRegSpectralClustering(
    int n_clusters, int num_samples, int num_features, int random_state,
    int info_view, int max_iter, int n_init, std::string affinity,
    int n_neighbors, double gamma, bool auto_num_clusters)
    : mvlearn::cluster::MVSpectralClustering(
          n_clusters, num_samples, num_features, random_state, info_view,
          max_iter, n_init, affinity, n_neighbors, gamma, auto_num_clusters){};

void MVCoRegSpectralClustering::fit(const std::vector<Eigen::MatrixXd>& Xs) {
  n_views_ = Xs.size();

  std::vector<Eigen::MatrixXd> check_u_mats(
      max_iter_, Eigen::MatrixXd(num_samples_, n_clusters_));

  // Compute the similarity matrices W_v for each of the views
  // The affinity matrix for each view is of size num_samples_ x num_samples_
  std::vector<Eigen::MatrixXd> sims(
      n_views_, Eigen::MatrixXd(num_samples_, num_samples_));
  int num_clusters_info_view{};
  fit_init_(Xs, sims, num_clusters_info_view);

  std::vector<int> idx_views(n_views_);
  std::iota(idx_views.begin(), idx_views.end(), 0);

  // Initialize matrices of eigenvectors
  std::vector<Eigen::MatrixXd> U_mats(
      n_views_, Eigen::MatrixXd(num_samples_, n_clusters_));
  std::vector<Eigen::MatrixXd> L_mats(
      n_views_, Eigen::MatrixXd(num_samples_, num_samples_));
  Eigen::MatrixXd obj_vals = Eigen::MatrixXd::Zero(n_views_, max_iter_);

  std::for_each(idx_views.begin(), idx_views.end(),
                [this, &sims = std::as_const(sims), &U_mats, &L_mats,
                 &obj_vals](const int& view) {
                  double o_val{};
                  computeEigs_(sims[view],    // X
                               n_clusters_,   // num_top_eigenvectors
                               U_mats[view],  // u_mat
                               L_mats[view],  // laplacian
                               o_val          // obj_val
                  );

                  obj_vals(view, 0) = o_val;
                });

  check_u_mats[0] = U_mats[0];

  // Iteratively solve for all U's
  int n_items = num_samples_;

  double v_lambda_ = 0.5;

  Eigen::MatrixXd l_comp = Eigen::MatrixXd(n_items, n_items);
  Eigen::MatrixXd l_mat = Eigen::MatrixXd(num_samples_, num_samples_);

  Eigen::MatrixXd u_mat = Eigen::MatrixXd(num_samples_, n_clusters_);
  Eigen::VectorXd d_mat =
      Eigen::VectorXd(n_clusters_);  // Notation, for consistency
  Eigen::MatrixXd laplacian = Eigen::MatrixXd(num_samples_, num_samples_);
  double o_val{};

  for (int it = 1; it < max_iter_; it++) {
    // Performing alternating maximization by cycling through all pairs of views
    // and udpating all except view 1

    for (int v1 = 1; v1 < n_views_; v1++) {
      // Computing the regularization term for view v1
      l_comp.setZero();
      for (int v2 = 0; v2 < n_views_; v2++) {
        if (v1 != v2) {
          l_comp += U_mats[v2] * U_mats[v2].transpose();
        }
        l_comp = 0.5 * (l_comp + l_comp.transpose());

        // Adding the symmetrized graph Laplacian for view v1
        l_mat = L_mats[v1] + v_lambda_ * l_comp;
      }

      computeEigs_(l_mat, n_clusters_, u_mat, laplacian, o_val);

      U_mats[v1] = u_mat;
      obj_vals(v1, it) = o_val;
    }

    // Update U and the objective function value for view 1
    l_comp.setZero();
    for (int vi = 1; vi < n_views_; vi++) {
      l_comp += U_mats[vi] * U_mats[vi].transpose();
    }
    l_comp = 0.5 * (l_comp + l_comp.transpose());
    l_mat = L_mats[0] + v_lambda_ * l_comp;

    computeEigs_(l_mat, n_clusters_, u_mat, laplacian, o_val);
    U_mats[0] = u_mat;
    obj_vals(0, it) = o_val;

    check_u_mats[it] = U_mats[0];
  }

  // Row normalize
  for (int view = 0; view < n_views_; view++) {
    for (int j = 0; j < U_mats[view].rows(); j++) {
      U_mats[view].row(j).normalize();
    }
  }

  // Perform k-means clustering
  sklearn::cluster::KMeans kmeans(n_clusters_, max_iter_);
  embedding_ = U_mats[info_view_];
  labels_ = kmeans.fit_predict(embedding_);
}

Eigen::VectorXi mvlearn::cluster::MVCoRegSpectralClustering::fit_predict(
    const std::vector<Eigen::MatrixXd>& Xs) {
  fit(Xs);

  return labels_;
}

}  // namespace mvlearn::cluster
