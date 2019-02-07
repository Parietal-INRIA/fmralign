import numpy as np
from sklearn.cluster import MiniBatchKMeans


def piecewise_transform(labels, estimators, X):
    """ piecewise transformation"""
    X_transform = np.zeros_like(X)
    for i in np.unique(labels):
        X_transform[labels == i] = estimators[i].transform(X[labels == i].T).T
    return X_transform


def hierarchical_k_means(X, n_clusters):
    """ use a recursive k-means to cluster X"""

    n_big_clusters = int(np.sqrt(n_clusters))
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_big_clusters, batch_size=1000,
                          n_init=10, max_no_improvement=10, verbose=0,
                          random_state=0).fit(X)
    coarse_labels = mbk.labels_
    fine_labels = np.zeros_like(coarse_labels)
    q = 0
    for i in range(n_big_clusters):
        n_small_clusters = int(
            n_clusters * np.sum(coarse_labels == i) * 1. / X.shape[0])
        n_small_clusters = np.maximum(1, n_small_clusters)
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_small_clusters,
                              batch_size=1000, n_init=10, max_no_improvement=10, verbose=0,
                              random_state=0).fit(X[coarse_labels == i])
        fine_labels[coarse_labels == i] = q + mbk.labels_
        q += n_small_clusters

    def _remove_empty_labels(labels):
        vals = np.unique(labels)
        inverse_vals = - np.ones(labels.max() + 1).astype(np.int)
        inverse_vals[vals] = np.arange(len(vals))
        return inverse_vals[labels]

    return _remove_empty_labels(fine_labels)


def load_img(masker, file_array, axis=0, confound=None):
    X = masker.transform(file_array,
                         confound)
    if type(X) == list:
        X = np.concatenate(X, axis=axis)
    X = X.T
    return X, masker
