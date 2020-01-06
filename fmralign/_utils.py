import numpy as np
from scipy.stats import pearsonr
from nilearn.regions.parcellations import Parcellations
from nilearn.image import smooth_img
from nilearn.masking import _apply_mask_fmri
from nilearn._utils.niimg_conversions import _check_same_fov
import nilearn
import nibabel
from packaging import version
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.base import TransformerMixin, ClusterMixin
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


def piecewise_transform(labels, estimators, X):
    """ Apply a piecewise transform to X:
    Parameters
    ----------
    labels: list of ints (len n_features)
        Parcellation of features in clusters
    estimators: list of estimators with transform() method
        I-th estimator will be applied on the i-th cluster of features
    X: nd array (n_samples, n_features)
        Data to transform

    Returns
    -------
    X_transform: nd array (n_features, n_samples)
        Transformed data
    """

    X_transform = np.zeros_like(X)
    # Labels are from 1 to n where as estimators are indexed from 0 to n-1
    for i in np.unique(labels):
        X_transform[:, labels == i] = estimators[i - 1].transform(
            X[:, labels == i])
    return X_transform


def _remove_empty_labels(labels):
    '''Remove empty values label values from labels list'''
    vals = np.unique(labels)
    inverse_vals = - np.ones(labels.max() + 1).astype(np.int)
    inverse_vals[vals] = np.arange(len(vals))
    return inverse_vals[labels]


def _hierarchical_k_means(X, n_clusters, init="k-means++", batch_size=1000,
                          n_init=10, max_no_improvement=10, verbose=0, random_state=0):
    """ Use a recursive k-means to cluster X
    Parameters
    ----------
    X: nd array (n_samples, n_features)
        Data to cluster

    n_clusters: int,
        The number of clusters to find.

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    batch_size : int, optional, default: 100
        Size of the mini batches. (Kmeans performed through MiniBatchKMeans)

    n_init : int, default=3
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the
        best of the ``n_init`` initializations as measured by inertia.

    max_no_improvement : int, default: 10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.
        To disable convergence detection based on inertia, set
        max_no_improvement to None.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    labels : list of ints (len n_features)
        Parcellation of features in clusters
    """

    n_big_clusters = int(np.sqrt(n_clusters))
    mbk = MiniBatchKMeans(init=init, n_clusters=n_big_clusters, batch_size=batch_size,
                          n_init=n_init, max_no_improvement=max_no_improvement, verbose=verbose,
                          random_state=random_state).fit(X)
    coarse_labels = mbk.labels_
    fine_labels = np.zeros_like(coarse_labels)
    q = 0
    for i in range(n_big_clusters):
        n_small_clusters = int(
            n_clusters * np.sum(coarse_labels == i) * 1. / X.shape[0])
        n_small_clusters = np.maximum(1, n_small_clusters)
        mbk = MiniBatchKMeans(init=init, n_clusters=n_small_clusters,
                              batch_size=batch_size, n_init=n_init,
                              max_no_improvement=max_no_improvement, verbose=verbose,
                              random_state=random_state).fit(X[coarse_labels == i])
        fine_labels[coarse_labels == i] = q + mbk.labels_
        q += n_small_clusters

    return _remove_empty_labels(fine_labels)


def _make_parcellation(imgs, clustering, n_pieces, masker, smoothing_fwhm=5, verbose=0):
    """Convenience function to use nilearn Parcellation class in our pipeline.
    It is used to find local regions of the brain in which alignment will be later applied.
    For alignment computational efficiency, regions should be of hundreds of voxels.

    Parameters
    ----------
    imgs: Niimgs
        data to cluster
    clustering: string or 3D Niimg
        In : {'kmeans', 'ward', 'rena'}, passed to nilearn Parcellations class.
        If you aim for speed, choose k-means (and check kmeans_smoothing_fwhm parameter)
        If you want spatially connected and/or reproducible regions use 'ward'
        If you want balanced clusters (especially from timeseries) used 'hierarchical_kmeans'
        For 'rena', need nilearn > 0.5.2
        If 3D Niimg, image used as predefined clustering, n_pieces is ignored
    n_pieces: int
        number of different labels
    masker: instance of NiftiMasker or MultiNiftiMasker
        Masker to be used on the data. For more information see:
        http://nilearn.github.io/manipulating_images/masker_objects.html
    smoothing_fwhm: None or int
        By default 5mm smoothing will be applied before kmeans clustering to have
        more compact clusters (but this will not change the data later).
        To disable this option, this parameter should be None.

    Returns
    -------
    labels : list of ints (len n_features)
        Parcellation of features in clusters
    """
    print(clustering)
    if type(clustering) == nibabel.nifti1.Nifti1Image:
        # check image makes suitable labels, this will return friendly error message if needed
        _check_same_fov(masker.mask_img_, clustering)
        labels_img = clustering
    elif clustering == "hierarchical_kmeans":
        images_to_parcel = imgs
        if smoothing_fwhm is not None:
            images_to_parcel = smooth_img(imgs, smoothing_fwhm)
        X = masker.transform(images_to_parcel)
        labels_ = _hierarchical_k_means(
            X.T, n_clusters=n_pieces, verbose=verbose) + 1
        labels_img = masker.inverse_transform(labels_)
    elif clustering in ['kmeans', 'ward', 'rena']:
        if clustering == "kmeans" and smoothing_fwhm is not None:
            images_to_parcel = smooth_img(imgs, smoothing_fwhm)
        try:
            parcellation = Parcellations(method=clustering, n_parcels=n_pieces, mask=masker,
                                         scaling=False, n_iter=20, verbose=verbose)
        except TypeError:
            if clustering == "rena" and (version.parse(nilearn.__version__) <= version.parse("0.5.2")):
                raise InputError(
                    ('ReNA algorithm is only available in Nilearn version > 0.5.2. \
                     Your version is {}. If you want to use ReNA, please run "pip install nilearn --upgrade"'.format(nilearn.__version__)))
            else:
                parcellation = Parcellations(
                    method=clustering, n_parcels=n_pieces, mask=masker, verbose=verbose)
        parcellation.fit(imgs)
        labels_img = parcellation.labels_img_
    else:
        raise InputError(
            ('Clustering should be "kmeans", "ward", "rena", "hierarchical_kmeans", \
             or a 3D Niimg'))
    return _apply_mask_fmri(labels_img, masker.mask_img_).astype(int)
