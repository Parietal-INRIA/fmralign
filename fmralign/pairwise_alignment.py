# -*- coding: utf-8 -*-
"""Module for pairwise functional alignment."""

import warnings

import numpy as np
from joblib import Memory, Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from fmralign import alignment_methods
from fmralign._utils import _transform_one_img
from fmralign.preprocessing import ParcellationMasker


def fit_one_piece(X_i, Y_i, alignment_method):
    """Align source and target data in one piece i, X_i and Y_i, using
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
        'ridge_cv', 'diagonal'
        - or an instance of one of alignment classes
            (imported from fmralign.alignment_methods)
    Returns
    -------

    alignment_algo
        Instance of alignment estimator class fitted for X_i, Y_i
    """
    if alignment_method == "identity":
        alignment_algo = alignment_methods.Identity()
    elif alignment_method == "scaled_orthogonal":
        alignment_algo = alignment_methods.ScaledOrthogonalAlignment()
    elif alignment_method == "ridge_cv":
        alignment_algo = alignment_methods.RidgeAlignment()
    elif alignment_method == "optimal_transport":
        alignment_algo = alignment_methods.OptimalTransportAlignment()
    elif alignment_method == "diagonal":
        alignment_algo = alignment_methods.DiagonalAlignment()
    elif isinstance(
        alignment_method,
        (
            alignment_methods.Identity,
            alignment_methods.ScaledOrthogonalAlignment,
            alignment_methods.RidgeAlignment,
            alignment_methods.OptimalTransportAlignment,
            alignment_methods.DiagonalAlignment,
            alignment_methods.POTAlignment,
        ),
    ):
        alignment_algo = clone(alignment_method)

    if not np.count_nonzero(X_i) or not np.count_nonzero(Y_i):
        warn_msg = (
            "Empty parcel found. Please check overlap between "
            "provided mask and functional image. Returning "
            "Identity alignment for empty parcel"
        )
        warnings.warn(warn_msg)
        alignment_algo = alignment_methods.Identity()
    try:
        alignment_algo.fit(X_i, Y_i)
    except UnboundLocalError:
        warn_msg = (
            f"{alignment_method} is an unrecognized "
            "alignment method. Please provide a recognized "
            "alignment method."
        )
        raise NotImplementedError(warn_msg)
    return alignment_algo


class PairwiseAlignment(BaseEstimator, TransformerMixin):
    """Decompose the source and target images into regions and align
    corresponding regions independently."""

    def __init__(
        self,
        alignment_method="identity",
        n_pieces=1,
        clustering="kmeans",
        mask=None,
        smoothing_fwhm=None,
        standardize=False,
        detrend=False,
        target_affine=None,
        target_shape=None,
        low_pass=None,
        high_pass=None,
        t_r=None,
        memory=Memory(location=None),
        memory_level=0,
        n_jobs=1,
        verbose=0,
    ):
        """If n_pieces > 1, decomposes the images into regions and align each
        source/target region independantly.

        Parameters
        ----------
        alignment_method: string
            Algorithm used to perform alignment between X_i and Y_i :
            * either 'identity', 'scaled_orthogonal', 'optimal_transport',
            'ridge_cv', 'diagonal'
            * or an instance of one of alignment classes
            (imported from fmralign.alignment_methods)
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
            n_pieces is then ignored.
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
        """Fit data X and Y and learn transformation to map X to Y.

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
        self.parcel_masker = ParcellationMasker(
            n_pieces=self.n_pieces,
            clustering=self.clustering,
            mask=self.mask,
            smoothing_fwhm=self.smoothing_fwhm,
            standardize=self.standardize,
            detrend=self.detrend,
            low_pass=self.low_pass,
            high_pass=self.high_pass,
            t_r=self.t_r,
            target_affine=self.target_affine,
            target_shape=self.target_shape,
            memory=self.memory,
            memory_level=self.memory_level,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        parceled_source, parceled_target = self.parcel_masker.fit_transform(
            [X, Y]
        )
        self.masker = self.parcel_masker.masker_
        self.mask = self.parcel_masker.masker_.mask_img_
        self.labels_ = self.parcel_masker.labels
        self.n_pieces = self.parcel_masker.n_pieces

        self.fit_ = Parallel(
            self.n_jobs, prefer="threads", verbose=self.verbose
        )(
            delayed(fit_one_piece)(X_i, Y_i, self.alignment_method)
            for X_i, Y_i in zip(
                parceled_source.to_list(), parceled_target.to_list()
            )
        )

        return self

    def transform(self, img):
        """Predict data from X.

        Parameters
        ----------
        img: Niimg-like object
            Source data

        Returns
        -------
        transformed_img: Niimg-like object
            Predicted data
        """
        if not hasattr(self, "fit_"):
            raise ValueError(
                "This instance has not been fitted yet. "
                "Please call 'fit' before 'transform'."
            )
        parceled_data_list = self.parcel_masker.transform(img)
        transformed_img = Parallel(
            self.n_jobs, prefer="threads", verbose=self.verbose
        )(
            delayed(_transform_one_img)(parceled_data, self.fit_)
            for parceled_data in parceled_data_list
        )
        if len(transformed_img) == 1:
            return transformed_img[0]
        else:
            return transformed_img

    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here.

        Will raise AttributeError if called.
        """
        raise AttributeError(
            "type object 'PairwiseAlignment' has no attribute 'fit_transform'"
        )

    def get_parcellation(self):
        """Get the parcellation masker used for alignment.

        Returns
        -------
        labels: `list` of `int`
            Labels of the parcellation masker.
        parcellation_img: Niimg-like object
            Parcellation image.
        """
        if hasattr(self, "parcel_masker"):
            check_is_fitted(self)
            labels = self.parcel_masker.get_labels()
            parcellation_img = self.parcel_masker.get_parcellation_img()
            return labels, parcellation_img
        else:
            raise AttributeError(
                (
                    "Parcellation has not been computed yet,"
                    "please fit the alignment estimator first."
                )
            )
