# -*- coding: utf-8 -*-
"""Module for pairwise functional alignment."""

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from fmralign import alignment_methods
from fmralign._utils import _transform_one_img, get_modality_features
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
        masker=None,
        modality="response",
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
        masker : None or :class:`~nilearn.maskers.NiftiMasker` or \
                :class:`~nilearn.maskers.MultiNiftiMasker`, or \
                :class:`~nilearn.maskers.SurfaceMasker` , optional
            A mask to be used on the data. If provided, the mask
            will be used to extract the data. If None, a mask will
            be computed automatically with default parameters.
        modality : str, optional (default='response')
            Specifies the alignment modality to be used:
            * 'response': Aligns by directly comparing corresponding similar 
            time points in the source and target images.
            * 'connectivity': Aligns based on voxel-wise connectivity features 
            within each parcel, comparing how each voxel relates to others in 
            the same region.
            * 'hybrid': Combines both time series and connectivity information 
            to perform the alignment.
        n_jobs: integer, optional (default = 1)
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        verbose: integer, optional (default = 0)
            Indicate the level of verbosity. By default, nothing is printed.
        """
        self.n_pieces = n_pieces
        self.alignment_method = alignment_method
        self.clustering = clustering
        self.masker = masker
        self.modality = modality
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
            masker=self.masker,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        self.parcel_masker.fit([X, Y])
        self.masker = self.parcel_masker.masker
        self.labels_ = self.parcel_masker.labels
        self.n_pieces = self.parcel_masker.n_pieces
        parcellation_img = self.parcel_masker.get_parcellation_img()

        # Add new features based on the modality
        X_, Y_ = get_modality_features(
            [X, Y], parcellation_img, self.masker, self.modality
        )

        # Parcelate the data
        parceled_source, parceled_target = self.parcel_masker.transform(
            [X_, Y_]
        )

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
