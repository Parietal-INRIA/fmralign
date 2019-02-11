import numpy as np
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.externals.joblib import Memory


def load_img(masker, imgs, axis=0, confound=None):
    """ Mask and concatenate imgs signal
    Parameters
    ----------
    masker: instance of NiftiMasker or MultiNiftiMasker
        Masker to be used on data.
    imgs: list of Niimg-like objects
        Data to be preprocessed See http://nilearn.github.io/manipulating_images/input_output.html
    axis: int
        if len(imgs)>1, axis along which imgs data is concatenated. If axis = 0, concatenation along sample axis. If axis = 1, concatenation along feature axis
    confound: list of confounds, optional
        List of confounds (2D arrays or filenames pointing to CSV files). Must be of same length than imgs. Passed to signal.clean. Please see the corresponding documentation for details.

    Returns
    -------
    X: nd array (n_features, n_samples)
        Masked and concatenated imgs data
    """
    X = masker.transform(imgs,
                         confound)
    if type(X) == list:
        X = np.concatenate(X, axis=axis)
    return X.T


def piecewise_transform(labels, estimators, X):
    """ Apply a piecewise transform to X:
    Parameters
    ----------
    labels: list of ints (len n_features)
        Parcellation of features in clusters
    estimators: list of estimators with transform() method
        I-th estimator will be applied on the i-th cluster of features
    X: nd array (n_features, n_samples)
        Data to transform

    Returns
    -------
    X_transform: nd array (n_features, n_samples)
        Transformed data
    """

    X_transform = np.zeros_like(X)
    for i in np.unique(labels):
        X_transform[labels == i] = estimators[i].transform(X[labels == i].T).T
    return X_transform


def _remove_empty_labels(labels):
    '''Remove empty values label values from labels list'''
    vals = np.unique(labels)
    inverse_vals = - np.ones(labels.max() + 1).astype(np.int)
    inverse_vals[vals] = np.arange(len(vals))
    return inverse_vals[labels]


def hierarchical_k_means(X, n_clusters):
    """ Use a recursive k-means to cluster X
    Parameters
    ----------
    X: nd array (n_samples, n_features)
        Data to cluster
    n_clusters: int
        Number of clusters to output
    Returns
    -------
    labels : list of ints (len n_features)
        Parcellation of features in clusters
    """

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

    return _remove_empty_labels(fine_labels)


def make_parcellation(X, mask, n_pieces, clustering_method='k_means', memory=Memory(cachedir=None)):
    """Separates input data into pieces

    Parameters
    ----------
    X: numpy array of shape (n_samples, n_features)
        data to cluster
    mask: numpy array
        the mask from which X is produced
    n_pieces: int
        number of different labels
    clustering_method: string, optional
        type of clustering applied to input data. Can be 'k_means', 'ward'

    Returns
    -------
    labels : list of ints (len n_features)
        Parcellation of features in clusters
    """
    if clustering_method == 'k_means':
        labels = hierarchical_k_means(X, n_pieces)
    elif clustering_method is 'ward':
        shape = mask.shape
        connectivity = grid_to_graph(*shape, mask=mask).tocsc()
        ward = AgglomerativeClustering(
            n_clusters=n_pieces, connectivity=connectivity, memory=memory)
        ward.fit(X)
        labels = ward.labels_
    return labels
