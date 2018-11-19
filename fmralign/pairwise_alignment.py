import copy
import numpy as np
from time import time

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from joblib import Parallel, delayed
from sklearn.externals.joblib import Memory
from sklearn.model_selection import ShuffleSplit
from nilearn.input_data.masker_validation import check_embedded_nifti_masker

from fmralign.alignment_methods import ScaledOrthogonalAlignment, RidgeAlignment, Identity, Hungarian
from fmralign._utils import hierarchical_k_means, piecewise_transform, load_img


def make_parcellation(X, mask, n_pieces, clustering_method='k_means', memory=Memory(cachedir=None)):
    """
    Separates input data into pieces

    Parameters
    ----------
    X: numpy array of shape (n_samples, n_features)
        input data
    mask: numpy array
        the mask from which X is produced
    n_pieces: int
        number of different labels
    clustering_method: string, optional
        type of clustering applied to input data. Can be 'k_means', 'ward'

    Returns
    -------
    labels: numpy array of shape (n_samples)
        labels[i] is the label of array i
        (0 <= labels[i] < n_samples)
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


def generate_Xi_Yi(labels, X, Y):
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(counts)
    for k in range(len(unique_labels)):
        label = unique_labels[k]
        i = label == labels
        if (k + 1) % 25 == 0:
            print("Fitting parcel: " + str(k + 1) +
                  "/" + str(len(unique_labels)))
        yield X[i], Y[i]


def fit_one_piece(X_i, Y_i, alignment_method):
    """
    Align source and target data in one piece i, X_i and Y_i, using alignment method and learn transformation to map X to Y.

    Parameters
    ----------
    X_i: ndarray
        Source data for piece i (shape : n_timeframes, n_i_voxels)
    Y_i: ndarray
        Target data for piece i (shape : n_timeframes, n_i_voxels)
    alignment_method: string
        algorithm used to perform alignment between X_i and Y_i can be taken
        'identity', 'scaled_orthogonal', 'ridge_cv', 'permutation' or an instance of one of alignment classes provided in functional_alignment.alignment_methods

    Returns
    -------
    alignment_algo
        Instance of alignment estimator class fitted for X_i, Y_i
    """

    if alignment_method == 'identity':
        alignment_algo = Identity()
    elif alignment_method == 'scaled_orthogonal':
        alignment_algo = ScaledOrthogonalAlignment()
    elif alignment_method == 'ridge_cv':
        alignment_algo = RidgeAlignment()
    elif alignment_method == 'permutation':
        alignment_algo = Hungarian()
    elif isinstance(alignment_method, (Identity, ScaledOrthogonalAlignment, RidgeAlignment, Hungarian)):
        alignment_algo = copy.deepcopy(alignment_method)
    alignment_algo.fit(X_i.T, Y_i.T)

    return alignment_algo


def fit_one_parcellation(X_, Y_, alignment_method, mask, n_pieces, clustering_method, joint_clustering, clustering_index, mem,
                         n_jobs=1, verbose=0):
    """
    Create one parcellation of n_pieces and align each source and target data in one piece i, X_i and Y_i, using alignment method and learn transformation to map X to Y.

    Parameters
    ----------
    X_: ndarray
        Source data (shape : n_timeframes, n_voxels)
    Y_: ndarray
        Target data (shape : n_timeframes, n_voxels)
    alignment_method: string
        algorithm used to perform alignment between each region of X_ and Y_
    mask: Niimg-like object
        Mask to be used on data.
    clustering_method : string
        method used to perform parcellation of data
    joint_clustering : boolean
        If True, clustering of voxels is performed jointly using X_ and Y_ signal. If false, it's performed only on X_
    clustering_index : list of integers
        Clustering is performed on a 20% subset of the data chosen randomly in timeframes. This index carry this subset.

    Returns
    -------
    alignment_algo
        Instance of alignment estimator class fitted for X_i, Y_i
    """
    if n_pieces > 1:
        if joint_clustering:
            clustering_data = np.hstack(
                X_[:, clustering_index], Y_[:, clustering_index])
        else:
            clustering_data = X_[:, clustering_index]
        labels = make_parcellation(clustering_data, mask,
                                   n_pieces, clustering_method, memory=mem)
    else:
        labels = np.zeros(int(mask.sum()), dtype=np.int8)

    fit = Parallel(n_jobs, backend="threading", verbose=verbose)(
        delayed(fit_one_piece)(
            X_i, Y_i, alignment_method
        ) for X_i, Y_i in generate_Xi_Yi(labels, X_, Y_)
    )

    return labels, fit


class PairwiseAlignment(BaseEstimator, TransformerMixin):
    """
    Decompose the source and target images into source and target regions
     Use alignment algorithms to align source and target regions independantly.
    """

    def __init__(self, alignment_method, n_pieces=1, clustering_method='k_means', joint_clustering=False, n_bags=1, perturbation=False, mask=None, smoothing_fwhm=None, standardize=None, detrend=False, target_affine=None, target_shape=None, low_pass=None, high_pass=None, t_r=None, memory=Memory(cachedir=None), memory_level=0, n_jobs=1, verbose=0):
        """
        Use alignment algorithms to align source and target images.
        If n_pieces > 1, decomposes the images into regions and align each source/target region independantly.
        If n_bags > 1, this parcellation process is applied multiple time and the resulting models are bagged.

        Parameters
        ----------
        alignment_method: string
        algorithm used to perform alignment between regions
        'identity', 'scaled_orthogonal', 'ridge_cv', 'permutation' or an instance of one of alignment classes provided in functional_alignment.alignment_methods
        n_pieces: int, optional
            number of regions in which the data is parcellated for alignment
            if 1 the alignment is done on full scale data, if >1, the voxels are clustered and alignment is performed on each cluster applied to X and Y.
        clustering_method : string, optional
            'ward' or 'k_means', method used for clustering of voxels
        joint_clustering : bool, optional
            if True, the clustering is done jointly on X and Y, if False it is done only using X signal
        n_bags: int, optional
            For parcellation alignments, number of bagged parcellations used.
        perturbation: bool, optional
            If true, the penalized transformation is the one that regress
             the source X to the difference between the target and the source Y - X.
            If false, the penalized transformaion is the one that regress the source X to
            the target Y
        mask: Niimg-like object, instance of       NiftiMasker or MultiNiftiMasker, optional
            Mask to be used on data. If an instance of masker is passed,
            then its mask will be used. If no mask is given,
            it will be computed automatically by a MultiNiftiMasker with default
            parameters.
        smoothing_fwhm: float, optional
            If smoothing_fwhm is not None, it gives the size in millimeters of the
            spatial smoothing to apply to the signal.
        standardize : boolean, optional
            If standardize is True, the time-series are centered and normed:
            their variance is put to 1 in the time dimension.
        detrend : boolean, optional
            This parameter is passed to nilearn.signal.clean. Please see the related documentation for details
        target_affine: 3x3 or 4x4 matrix, optional
            This parameter is passed to nilearn.image.resample_img. Please see the
            related documentation for details.
        target_shape: 3-tuple of integers, optional
            This parameter is passed to nilearn.image.resample_img. Please see the
            related documentation for details.
        low_pass: None or float, optional
            This parameter is passed to nilearn.signal.clean. Please see the related
            documentation for details
        high_pass: None or float, optional
            This parameter is passed to nilearn.signal.clean. Please see the related
            documentation for details
        t_r: float, optional
            This parameter is passed to nilearn.signal.clean. Please see the related
            documentation for details
        memory: instance of joblib.Memory or string
            Used to cache the masking process and results of algorithms.
            By default, no caching is done. If a string is given, it is the
            path to the caching directory.
        memory_level: integer, optional
            Rough estimator of the amount of memory used by caching. Higher value
            means more memory for caching.
        n_jobs: integer, optional
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        verbose: integer, optional
            Indicate the level of verbosity. By default, nothing is printed.
        """
        self.n_pieces = n_pieces
        self.alignment_method = alignment_method
        self.perturbation = perturbation
        self.n_bags = n_bags
        self.clustering_method = clustering_method
        self.joint_clustering = joint_clustering
        self.mask = mask
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, Y):
        """
        Fit data X and Y and learn transformation to map X to Y
        Parameters
        ----------
        X: Niimg-like object
           See http://nilearn.github.io/manipulating_images/input_output.html
           source data
        Y: Niimg-like object
           See http://nilearn.github.io/manipulating_images/input_output.html
           target data

        Returns
        -------
        self
        """
        self.masker_ = check_embedded_nifti_masker(self)
        self.masker_.n_jobs = 1  # self.n_jobs
        # Avoid warning with imgs != None
        # if masker_ has been provided a mask_img
        if self.masker_.mask_img is None:
            self.masker_.fit([X])
        else:
            self.masker_.fit()
        X_, _ = load_img(self.masker_, X)
        Y_, _ = load_img(self.masker_, Y)

        if self.perturbation:
            Y_ = Y_ - X_

        self.fit_, self.labels_ = [], []
        rs = ShuffleSplit(n_splits=self.n_bags,
                          test_size=.8, random_state=0)

        outputs = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_one_parcellation)(
                X_, Y_, self.alignment_method, self.masker_.mask_img.get_data(), self.n_pieces, self.clustering_method, self.joint_clustering, clustering_index, self.memory, self.n_jobs, verbose=self.verbose) for clustering_index, _ in rs.split(Y_.T))

        self.labels_ = [output[0] for output in outputs]
        self.fit_ = [output[1] for output in outputs]

        return self

    def transform(self, X):
        """
        Predict data from X
        Parameters
        ----------
        X: Niimg-like object
           See http://nilearn.github.io/manipulating_images/input_output.html
           source data

        Returns
        -------
        X_transform: Niimg-like object
           See http://nilearn.github.io/manipulating_images/input_output.html
           predicted data
        """
        X_, _ = load_img(self.masker_, X)

        X_transform = np.zeros_like(X_)
        for i in range(self.n_bags):
            X_transform += piecewise_transform(
                self.labels_[i], self.fit_[i], X_)

        X_transform /= self.n_bags

        if self.perturbation:
            X_transform += X_

        return self.masker_.inverse_transform(X_transform.T)
