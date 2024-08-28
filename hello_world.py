import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy


data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

nbrs = NearestNeighbors(n_neighbors=3)
nbrs.fit(data)
distances, indices = nbrs.kneighbors(data)
sims = nbrs.kneighbors_graph(data).toarray()


X = [
    [
        0.178395,
        0.367737,
        0.655419,
        0.41206,
        0.576451,
        0.485838,
        0.879841,
        0.182721,
        0.578371,
        0.554799,
    ],
    [
        0.399677,
        0.222367,
        0.527644,
        0.81593,
        0.757953,
        0.0252066,
        0.0174133,
        0.472538,
        0.703867,
        0.793983,
    ],
    [
        0.166599,
        0.219657,
        0.505857,
        0.280523,
        0.176208,
        0.69734,
        0.564316,
        0.650993,
        0.974248,
        0.783903,
    ],
    [
        0.212122,
        0.201738,
        0.915692,
        0.578658,
        0.796108,
        0.991695,
        0.160365,
        0.35893,
        0.275711,
        0.119115,
    ],
    [
        0.0619357,
        0.0441157,
        0.188085,
        0.0280522,
        0.959691,
        0.940898,
        0.596072,
        0.268646,
        0.695562,
        0.954347,
    ],
    [
        0.0541498,
        0.34392,
        0.119156,
        0.342459,
        0.220324,
        0.885425,
        0.592369,
        0.610684,
        0.915147,
        0.379974,
    ],
    [
        0.275665,
        0.647654,
        0.0580968,
        0.632808,
        0.140028,
        0.110851,
        0.502824,
        0.579253,
        0.161136,
        0.711484,
    ],
    [
        0.0477572,
        0.149464,
        0.719929,
        0.303718,
        0.607345,
        0.998995,
        0.22888,
        0.408675,
        0.806413,
        0.457171,
    ],
    [
        0.321033,
        0.0296677,
        0.637535,
        0.390216,
        0.369788,
        0.605354,
        0.896086,
        0.218028,
        0.914142,
        0.608854,
    ],
    [
        0.272734,
        0.878494,
        0.880846,
        0.953841,
        0.169696,
        0.748386,
        0.89304,
        0.949041,
        0.766489,
        0.60757,
    ],
]

X = np.transpose(X) @ X
n_clusters = 5

d_mat = np.diag(np.sum(X, axis=1))

# Double check why we take absolute value of d_mat
d_alt = np.sqrt(np.linalg.inv(np.abs(d_mat)))
laplacian = d_alt @ X @ d_alt

# Make the resulting matrix symmetric
laplacian = (laplacian + np.transpose(laplacian)) / 2.0

# To ensure PSD
min_val = laplacian.min()
if min_val < 0:
    laplacian = laplacian + min_val

# Obtain the top n_cluster eigenvectors of the laplacian
v_vals, u_mat = scipy.linalg.eigh(laplacian, driver="evd")
la_eigs = u_mat[:, (u_mat.shape[1] - n_clusters) : u_mat.shape[1]]
