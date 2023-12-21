from numpy.random.mtrand import RandomState
from mvlearn.cluster.mv_coreg_spectral import MultiviewCoRegSpectralClustering
import mvlearnpycpp
import numpy as np

import matplotlib.pyplot as plt
import time

import mvlearn
from mvlearn.datasets import load_UCImultifeature
from mvlearn.cluster import MultiviewSpectralClustering

from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.datasets import make_moons
from mvlearn.plotting import quick_visualize

import pandas as pd

RANDOM_SEED = 123456


def display_plots(pre_title, data, labels):
    # plot the views
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    dot_size = 10
    ax[0].scatter(data[0][:, 0], data[0][:, 1], c=labels, s=dot_size)
    ax[0].set_title(pre_title + " View 1")
    ax[0].axes.get_xaxis().set_visible(False)
    ax[0].axes.get_yaxis().set_visible(False)

    ax[1].scatter(data[1][:, 0], data[1][:, 1], c=labels, s=dot_size)
    ax[1].set_title(pre_title + " View 2")
    ax[1].axes.get_xaxis().set_visible(False)
    ax[1].axes.get_yaxis().set_visible(False)

    plt.show()


# A function to generate the moons data
def create_moons(seed, num_per_class=500, num_views=2, noise=0.03, shuffle=False):
    np.random.seed(seed)
    data = []
    labels = []

    for view in range(num_views):
        v_dat, v_labs = make_moons(
            num_per_class * 2, random_state=seed + view, noise=noise, shuffle=shuffle
        )
        if view == num_views - 1:
            v_dat = v_dat[:, ::-1]

        data.append(v_dat)
    for ind in range(len(data)):
        labels.append(
            ind
            * np.ones(
                num_per_class,
            )
        )
    labels = np.concatenate(labels)

    return data, labels


# Generating the data
m_data, labels = create_moons(RANDOM_SEED)
n_class = 2


gamma = 0.5
gamma_mpp = 0.001
max_iter = 22

affinity = "rbf"
# affinity = "nearest_neighbors"
# affinity_mpp = "rbf_constant_scale"
affinity_mpp = "rbf_local_scale_l2"
# affinity = "rbf"
# affinity_mpp = "rbf"
n_neighbors = 10
info_view = 0
use_spectra = True

# num_samples = m_data[0].shape[0]
# num_features = m_data[0].shape[1]

data0 = pd.read_csv("./test_data0.csv")
data0 = data0.iloc[:, 1:]
data1 = pd.read_csv("./test_data1.csv")
data1 = data1.iloc[:, 1:]
data0 = data0.to_numpy()
data1 = data1.to_numpy()
data0 = np.transpose(data0)
data1 = np.transpose(data1)

m_data = [data0, data1]
num_samples = m_data[0].shape[0]  # 453
num_features = m_data[0].shape[1]  # 252

n_clusters = 25
auto_num_clusters = True

t0 = time.time()
mvsc = mvlearnpycpp.MultiviewSpectralClustering(
    n_clusters,
    num_samples,  # Number of stocks
    num_features,  # Number of time stamps
    info_view,
    max_iter,
    affinity_mpp,
    n_neighbors,
    gamma_mpp,
    auto_num_clusters,
    use_spectra,
)
mpp_clusters = mvsc.fit_predict(m_data)
t1 = time.time()
print("mv_clusters time {}".format(t1 - t0))


t0 = time.time()
mvcoregsc = mvlearnpycpp.MultiviewCoRegSpectralClustering(
    n_clusters,
    num_samples,
    num_features,
    info_view,
    max_iter,
    affinity_mpp,
    n_neighbors,
    gamma_mpp,
    auto_num_clusters,
    use_spectra,
)
mvcoregsc_clusters = mvcoregsc.fit_predict(m_data)
t1 = time.time()
print("m_coregclusters time {}".format(t1 - t0))

breakpoint()
# mcrpp_clusters = mvcoregsc.fit_predict(m_data)
# mcrpp_clusters = 1 - mcrpp_clusters
# t1 = time.time()
# print("w".format(t1 - t0))
# mcrpp_nmi = nmi_score(labels, mcrpp_clusters)
# print(f"mcrpp_nmi = {mcrpp_nmi:.3f}")
# print(f"mcrpp_num_clusters = {str(mvcoregsc.get_num_clusters())}")
#

# mvlearn
t0 = time.time()
m_spectral = MultiviewSpectralClustering(
    n_clusters=n_class,
    affinity=affinity,
    max_iter=max_iter,
    info_view=info_view,
    random_state=RANDOM_SEED,
    n_init=10,
    gamma=None,
    n_neighbors=n_neighbors,
)
m_clusters = m_spectral.fit_predict(m_data)

breakpoint()

# mvlearn
t0 = time.time()
m_coregspectral = MultiviewCoRegSpectralClustering(
    n_clusters=n_class,
    affinity=affinity,
    max_iter=max_iter,
    info_view=info_view,
    random_state=RANDOM_SEED,
    n_init=10,
    gamma=gamma,
    n_neighbors=n_neighbors,
)
m_coregclusters = m_coregspectral.fit_predict(m_data)
t1 = time.time()
print("m_coregclusters time {}".format(t1 - t0))
m_coregnmi = nmi_score(labels, m_coregclusters)
print(f"m_coregnmi = {m_coregnmi:.3f}")

# display_plots("Ground truth", m_data, labels)
# display_plots("mv clustering", m_data, m_clusters)
# display_plots("mvpp clustering", m_data, mpp_clusters)
