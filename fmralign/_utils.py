import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.externals.joblib import Memory
from nilearn.regions.parcellations import Parcellations
from nilearn.image import smooth_img
from nilearn.masking import _apply_mask_fmri


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


def make_parcellation(X, mask, n_pieces, clustering_method='k_means', memory=Memory(cachedir=None), to_filename=None):
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
    to_filename: str
        path to which the parcellation will be saved

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
    if to_filename is not None:
        labels_img_.to_filename(to_filename)
    return labels


def _make_parcellation(imgs, clustering_method, n_pieces, masker, to_filename=None, kmeans_smoothing_fwhm=5, verbose=0):
    """Convenience function to use nilearn Parcellation class in our pipeline.
    It is used to find local regions of the brain in which alignment will be later applied.
    For alignment computational efficiency, regions should be of hundreds of voxels.

    Parameters
    ----------
    imgs: Niimgs
        data to cluster
    clustering_method: string
        In : {'kmeans', 'ward', 'rena'}, passed to nilearn Parcellations class.
        If you aim for speed, choose k-means (and check kmeans_smoothing_fwhm parameter)
        If you want spatially connected and/or reproducible regions use 'ward'
    n_pieces: int
        number of different labels
    masker: instance of NiftiMasker or MultiNiftiMasker
    to_filename: str, optional
        path to which the parcellation will be saved
    kmeans_smoothing_fwhm: None or int
        By default 5mm smoothing will be applied before clusterisation to have
        more compact clusters (but this will not change the data later).
        To disable this option, this parameter should be None.

    Returns
    -------
    labels : list of ints (len n_features)
        Parcellation of features in clusters
    """
    if clustering_method == "kmeans" and kmeans_smoothing_fwhm is not None:
        images_to_parcel = smooth_img(imgs, kmeans_smoothing_fwhm)
    try:
        parcellation = Parcellations(method=clustering_method, n_parcels=n_pieces, mask=masker,
                                     scaling=False, n_iter=20, verbose=verbose)
        parcellation.fit()
    except TypeError:
        if clustering_method == "rena":
            raise InputError(
                ('ReNA algorithm is only available in Nilearn version > 0.5.2. If you want to use it, please run "pip install --upgrade nilearn"'))
        else:
            parcellation = Parcellations(
                method=clustering_method, n_parcels=n_pieces, mask=masker, verbose=verbose)
    parcellation.fit(imgs)
    if to_filename is not None:
        parcellation.labels_img_.to_filename(to_filename)
    return _apply_mask_fmri(parcellation.labels_img_, masker.mask_img_)


def voxelwise_correlation(ground_truth, prediction, masker):
    """
    Parameters
    ----------
    ground_truth: 3D or 4D Niimg
        Reference image (data acquired but never used before and considered as missing)
    prediction : 3D or 4D Niimg
        Same shape as ground_truth
    masker : instance of NiftiMasker
        masker to use on ground truth and prediction

    Returns
    -------
    voxelwise_correlation : 3D Niimg
        Voxelwise score between ground_truth and prediction
    """
    X_gt = masker.transform(ground_truth)
    X_pred = masker.transform(prediction)

    voxelwise_correlation = np.array([pearsonr(X_gt[:, vox], X_pred[:, vox])[0]
                                      for vox in range(X_pred.shape[1])])
    return masker.inverse_transform(voxelwise_correlation)
