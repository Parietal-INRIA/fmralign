# -*- coding: utf-8 -*-
""" Module for pairwise functional alignment
"""
import warnings
import numpy as np
import os

from sklearn.base import BaseEstimator, TransformerMixin
from joblib import (delayed, Memory, Parallel)
from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone
from nilearn.input_data.masker_validation import check_embedded_nifti_masker
from nilearn.image import load_img, concat_imgs, index_img
import nibabel as nib
from fmralign.alignment_methods import RidgeAlignment, Identity, Hungarian, \
    ScaledOrthogonalAlignment, OptimalTransportAlignment, DiagonalAlignment
from fmralign._utils import _make_parcellation, piecewise_transform, _intersect_clustering_mask


def generate_Xi_Yi(labels, X, Y, masker, verbose):
    """ Generate source and target data X_i and Y_i for each piece i.

    Parameters
    ----------
    labels : list of ints (len n_features)
        Parcellation of features in clusters
    X: Niimg-like object
        Source data
    Y: Niimg-like object
        Target data
    masker: instance of NiftiMasker or MultiNiftiMasker
        Masker to be used on the data. For more information see:
        http://nilearn.github.io/manipulating_images/masker_objects.html
    verbose: integer, optional.
        Indicate the level of verbosity.

    Yields
    -------
    X_i: ndarray
        Source data for piece i (shape : n_samples, n_features)
    Y_i: ndarray
        Target data for piece i (shape : n_samples, n_features)

    """
    X_ = masker.transform(X)
    Y_ = masker.transform(Y)
    unique_labels = np.unique(labels)

    for k in range(len(unique_labels)):
        label = unique_labels[k]
        i = label == labels
        if (k + 1) % 25 == 0 and verbose > 0:
            print("Fitting parcel: " + str(k + 1) +
                  "/" + str(len(unique_labels)))
        # should return X_i Y_i
        yield X_[:, i], Y_[:, i]


def fit_one_piece(X_i, Y_i, alignment_method):
    """ Align source and target data in one piece i, X_i and Y_i, using
    alignment method and learn transformation to map X to Y.

    Parameters
    ----------
    X_i: ndarray
        Source data for piece i (shape : n_samples, n_features)
    Y_i: ndarray
        Target data for piece i (shape : n_samples, n_features)
    alignment_method: string
        Algorithm used to perform alignment between X_i and Y_i :
        - either 'identity', 'scaled_orthogonal', 'optimal_transport',
        'ridge_cv', 'permutation', 'diagonal'
        - or an instance of one of alignment classes
            (imported from functional_alignment.alignment_methods)
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
    elif alignment_method == 'optimal_transport':
        alignment_algo = OptimalTransportAlignment()
    elif alignment_method == 'diagonal':
        alignment_algo = DiagonalAlignment()
    elif isinstance(alignment_method, (Identity, ScaledOrthogonalAlignment,
                                       RidgeAlignment, Hungarian,
                                       OptimalTransportAlignment,
                                       DiagonalAlignment)):
        alignment_algo = clone(alignment_method)

    if not np.count_nonzero(X_i) or not np.count_nonzero(Y_i):
        warn_msg = ("Empty parcel found. Please check overlap between " +
                    "provided mask and functional image. Returning " +
                    "Identity alignment for empty parcel")
        warnings.warn(warn_msg)
        alignment_algo = Identity()
    try:
        alignment_algo.fit(X_i, Y_i)
    except UnboundLocalError:
        warn_msg = ("{} is an unrecognized ".format(alignment_method) +
                    "alignment method. Please provide a recognized " +
                    "alignment method.")
        raise NotImplementedError(warn_msg)
    return alignment_algo


def fit_one_parcellation(X_, Y_, alignment_method, masker, n_pieces,
                         clustering, clustering_index,
                         n_jobs, verbose):
    """ Create one parcellation of n_pieces and align each source and target
    data in one piece i, X_i and Y_i, using alignment method
    and learn transformation to map X to Y.

    Parameters
    ----------
    X_: Niimg-like object
        Source data
    Y_: Niimg-like object
        Target data
    alignment_method: string
        algorithm used to perform alignment between each region of X_ and Y_
    masker: instance of NiftiMasker or MultiNiftiMasker
        Masker to be used on the data. For more information see:
        http://nilearn.github.io/manipulating_images/masker_objects.html
    n_pieces: n_pieces: int,
        Number of regions in which the data is parcellated for alignment
    clustering: string or 3D Niimg
        method used to perform parcellation of data.
        If 3D Niimg, image used as predefined clustering.
    clustering_index: list of integers
        Clustering is performed on a 20% subset of the data chosen randomly
        in timeframes. This index carry this subset.
    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.
    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    Returns
    -------
    alignment_algo
        Instance of alignment estimator class fitted for X_i, Y_i
    """
    # choose indexes maybe with index_img to not
    labels = _make_parcellation(X_, clustering_index, clustering,
                                n_pieces, masker, verbose=verbose)

    fit = Parallel(n_jobs, prefer="threads", verbose=verbose)(
        delayed(fit_one_piece)(
            X_i, Y_i, alignment_method
        ) for X_i, Y_i in generate_Xi_Yi(labels, X_, Y_, masker, verbose)
    )

    return labels, fit


class PairwiseAlignment(BaseEstimator, TransformerMixin):
    """
    Decompose the source and target images into regions and align corresponding \
    regions independently.
    """

    def __init__(self, alignment_method, n_pieces=1,
                 clustering='kmeans', n_bags=1, mask=None,
                 smoothing_fwhm=None, standardize=False, detrend=False,
                 target_affine=None, target_shape=None, low_pass=None,
                 high_pass=None, t_r=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0):
        """
        If n_pieces > 1, decomposes the images into regions \
        and align each source/target region independantly.
        If n_bags > 1, this parcellation process is applied multiple time \
        and the resulting models are bagged.

        Parameters
        ----------
        alignment_method: string
            Algorithm used to perform alignment between X_i and Y_i :
            * either 'identity', 'scaled_orthogonal', 'optimal_transport',
            'ridge_cv', 'permutation', 'diagonal'
            * or an instance of one of alignment classes
            (imported from functional_alignment.alignment_methods)
        n_pieces: int, optional (default = 1)
            Number of regions in which the data is parcellated for alignment.
            If 1 the alignment is done on full scale data.
            If >1, the voxels are clustered and alignment is performed
            on each cluster applied to X and Y.
        clustering : string or 3D Niimg optional (default : kmeans)
            'kmeans', 'ward', 'rena', 'hierarchical_kmeans' method used for
            clustering of voxels based on functional signal, passed to
            nilearn.regions.parcellations
            If 3D Niimg, image used as predefined clustering,
            n_bags and n_pieces are then ignored.
        n_bags: int, optional (default = 1)
            If 1 : one estimator is fitted.
            If >1 number of bagged parcellations and estimators used.
        mask: Niimg-like object, instance of NiftiMasker or
                                MultiNiftiMasker, optional (default = None)
            Mask to be used on data. If an instance of masker is passed,
            then its mask will be used. If no mask is given,
            it will be computed automatically by a MultiNiftiMasker
            with default parameters.
        smoothing_fwhm: float, optional (default = None)
            If smoothing_fwhm is not None, it gives the size in millimeters
            of the spatial smoothing to apply to the signal.
        standardize: boolean, optional (default = False)
            If standardize is True, the time-series are centered and normed:
            their variance is put to 1 in the time dimension.
        detrend: boolean, optional (default = None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details
        target_affine: 3x3 or 4x4 matrix, optional (default = None)
            This parameter is passed to nilearn.image.resample_img.
            Please see the related documentation for details.
        target_shape: 3-tuple of integers, optional (default = None)
            This parameter is passed to nilearn.image.resample_img.
            Please see the related documentation for details.
        low_pass: None or float, optional (default = None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details.
        high_pass: None or float, optional (default = None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details.
        t_r: float, optional (default = None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details.
        memory: instance of joblib.Memory or string (default = None)
            Used to cache the masking process and results of algorithms.
            By default, no caching is done. If a string is given, it is the
            path to the caching directory.
        memory_level: integer, optional (default = None)
            Rough estimator of the amount of memory used by caching.
            Higher value means more memory for caching.
        n_jobs: integer, optional (default = 1)
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        verbose: integer, optional (default = 0)
            Indicate the level of verbosity. By default, nothing is printed.
        """
        self.n_pieces = n_pieces
        self.alignment_method = alignment_method
        self.n_bags = n_bags
        self.clustering = clustering
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
        """Fit data X and Y and learn transformation to map X to Y

        Parameters
        ----------
        X: Niimg-like object
            Source data.

        Y: Niimg-like object
            Target data

        Returns
        -------
        self
        """
        self.masker_ = check_embedded_nifti_masker(self)
        self.masker_.n_jobs = self.n_jobs

        if self.masker_.mask_img is None:
            self.masker_.fit([X])
        else:
            self.masker_.fit()

        if type(self.clustering) == nib.nifti1.Nifti1Image or os.path.isfile(self.clustering):
            # check that clustering provided fills the mask, if not, reduce the mask
            if 0 in self.masker_.transform(self.clustering):
                reduced_mask = _intersect_clustering_mask(
                    self.clustering, self.masker_.mask_img)
                self.mask = reduced_mask
                self.masker_ = check_embedded_nifti_masker(self)
                self.masker_.n_jobs = self.n_jobs
                self.masker_.fit()
                warnings.warn(
                    "Mask used was bigger than clustering provided. " +
                    "Its intersection with the clustering was used instead.")

        if isinstance(X, (list, np.ndarray)):
            X_ = concat_imgs(X)
        else:
            X_ = load_img(X)
        if isinstance(X, (list, np.ndarray)):
            Y_ = concat_imgs(Y)
        else:
            Y_ = load_img(Y)

        self.fit_, self.labels_ = [], []
        rs = ShuffleSplit(n_splits=self.n_bags,
                          test_size=.8, random_state=0)

        outputs = Parallel(n_jobs=self.n_jobs, prefer="threads",
                           verbose=self.verbose)(
            delayed(fit_one_parcellation)(
                X_, Y_, self.alignment_method, self.masker_, self.n_pieces,
                self.clustering, clustering_index, self.n_jobs, self.verbose)
            for clustering_index, _ in rs.split(range(X_.shape[-1])))
        # change split
        self.labels_ = [output[0] for output in outputs]
        self.fit_ = [output[1] for output in outputs]

        return self

    def transform(self, X):
        """Predict data from X

        Parameters
        ----------
        X: Niimg-like object
            Source data

        Returns
        -------
        X_transform: Niimg-like object
            Predicted data
        """
        if isinstance(X, (list, np.ndarray)):
            X = concat_imgs(X)
        X_ = self.masker_.transform(X)

        X_transform = np.zeros_like(X_)
        for i in range(self.n_bags):
            X_transform += piecewise_transform(
                self.labels_[i], self.fit_[i], X_)

        X_transform /= self.n_bags

        return self.masker_.inverse_transform(X_transform)

    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here. Will raise AttributeError if called.
        """
        raise AttributeError(
            "type object 'PairwiseAlignment' has no attribute 'fit_transform'")
