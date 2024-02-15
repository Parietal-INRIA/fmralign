# -*- coding: utf-8 -*-
""" Test module to adapt quickly fmralign for piecewise srm
Implementation from fastSRM is taken from H. Richard
"""
# Author: T. Bazeille
# License: simplified BSD

import os
import warnings

import numpy as np
import nibabel as nib
from sklearn.base import clone
from joblib import delayed, Parallel
from nilearn.image import concat_imgs, load_img
from sklearn.model_selection import ShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from nilearn._utils.masker_validation import check_embedded_masker

from ._utils import _make_parcellation, _intersect_clustering_mask


class Identity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.basis_list = [np.eye(X[0].shape[0]) for _ in range(len(X))]
        self.basis_list = [np.eye(X[0].shape[0]) for _ in range(len(X))]
        return self

    def transform(self, X, y=None, subjects_indexes=None):
        if subjects_indexes is None:
            subjects_indexes = np.arange(len(self.basis_list))
        return np.array([X[i] for i in range(len(subjects_indexes))])

    def inverse_transform(self, X, subjects_indexes=None):
        return np.array([X for _ in subjects_indexes])

    def add_subjects(self, X_list, S):
        self.basis_list = [w for w in self.basis_list] + [
            np.eye(x.shape[0]) for x in X_list
        ]


def generate_X_is(labels, X_list, masker, verbose):
    """Generate source and target data X_i and Y_i for each piece i.

    Parameters
    ----------
    labels : list of ints (len n_features)
        Parcellation of features in clusters
    X_list: list of Niimgs
        Source data
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

    masked_X_list = []
    for X in X_list:
        masked_X_list.append(masker.transform(X))
    unique_labels = np.unique(labels)

    for k in range(len(unique_labels)):
        label = unique_labels[k]
        i = label == labels
        if (k + 1) % 25 == 0 and verbose > 0:
            print("Fitting parcel: " + str(k + 1) + "/" + str(len(unique_labels)))
        # should return X_i Y_i

        yield [X_[:, i].T for X_ in masked_X_list]


def fit_one_piece(piece_X_list, method):
    """Fit SRM on group source data in one piece i, piece_X_list, using
    alignment method.

    Parameters
    ----------
    piece_X_list: ndarray
        Source data for piece i (shape : n_subjects, n_samples, n_features)
    method: string
        Algorithm used to perform groupwise alignment on piece_X_list :
        - either 'identity',
        - or an instance of IdentifiableFastSRM alignment class (imported from fastsrm)
    Returns
    -------
    alignment_algo
        Instance of alignment estimator class fit on
    """
    if method == "identity":
        alignment_algo = Identity()
    else:
        from fastsrm.identifiable_srm import IdentifiableFastSRM

        # if isinstance(method, (FastSRM, MultiViewICA, AdaptiveMultiViewICA)):
        if isinstance(method, (IdentifiableFastSRM)):
            alignment_algo = clone(method)
            if hasattr(alignment_algo, "aggregate"):
                alignment_algo.aggregate = None
            if np.shape(piece_X_list)[1] < alignment_algo.n_components:
                alignment_algo.n_components = np.shape(piece_X_list)[1]
        else:
            warn_msg = (
                "Method not recognized, should be 'identity' "
                "or an instance of IdentifiableFastSRM"
            )
            NotImplementedError(warn_msg)

    # dirty monkey patching to avoid having n_components > n_voxels in any
    # piece which would yield a bug in add_subjects()
    reduced_SR = alignment_algo.fit(piece_X_list).transform(piece_X_list)

    if len(reduced_SR) == len(piece_X_list):
        reduced_SR = np.mean(reduced_SR, axis=0)

    return alignment_algo, reduced_SR


def fit_one_parcellation(
    X_list,
    srm,
    masker,
    n_pieces,
    clustering,
    clustering_index,
    n_jobs,
    verbose,
):
    """Create parcellation of n_pieces and align one piece i in group source
    data X_list, using SRM alignment instance.

    Parameters
    ----------
    X_list: Iterable of Niimg-like objects
        Source data
    srm : FastSRM instance
    masker: instance of NiftiMasker or MultiNiftiMasker
        Masker to be used on the data. For more information see:
        http://nilearn.github.io/manipulating_images/masker_objects.html
    n_pieces: integer
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
    labels = _make_parcellation(
        X_list[0],
        clustering_index,
        clustering,
        n_pieces,
        masker,
        verbose=verbose,
    )

    outputs = Parallel(n_jobs, prefer="threads", verbose=verbose)(
        delayed(fit_one_piece)(piece_X_list, srm)
        for piece_X_list in generate_X_is(labels, X_list, masker, verbose)
    )
    fit = [output[0] for output in outputs]
    reduced_sr = [output[1] for output in outputs]
    return labels, fit, reduced_sr


class PiecewiseModel(BaseEstimator, TransformerMixin):
    """
    Decompose the source images into regions and summarize subjects information
    in a SR, then use alignment to predict new contrast for target(s) subject.
    """

    def __init__(
        self,
        srm,
        n_pieces=1,
        clustering="kmeans",
        n_bags=1,
        mask=None,
        smoothing_fwhm=None,
        standardize=False,
        detrend=None,
        target_affine=None,
        target_shape=None,
        low_pass=None,
        high_pass=None,
        t_r=None,
        n_jobs=1,
        verbose=0,
        reshape=True,
    ):
        """
        Parameters
        ----------
        srm : FastSRM instance
        n_pieces: int, optional (default = 1)
            Number of regions in which the data is parcellated for alignment.
            If 1 the alignment is done on full scale data.
            If > 1, the voxels are clustered and alignment is performed on each
            cluster applied to X and Y.
        clustering : string or 3D Niimg optional (default : kmeans)
            'kmeans', 'ward', 'rena', 'hierarchical_kmeans' method used for
            clustering of voxels based on functional signal,
            passed to nilearn.regions.parcellations
            If 3D Niimg, image used as predefined clustering,
            n_bags and n_pieces are then ignored.
        n_bags: int, optional (default = 1)
            If 1 : one estimator is fitted.
            If >1 number of bagged parcellations and estimators used.
        mask: Niimg-like object, instance of NiftiMasker or
                                MultiNiftiMasker, optional (default = None)
            Mask to be used on data. If an instance of masker is passed, then its
            mask will be used. If no mask is given, it will be computed
            automatically by a MultiNiftiMasker with default parameters.
        smoothing_fwhm: float, optional (default = None)
            If smoothing_fwhm is not None, it gives the size in millimeters
            of the spatial smoothing to apply to the signal.
        standardize: boolean, optional (default = None)
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
        n_jobs: integer, optional (default = 1)
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        verbose: integer, optional (default = 0)
            Indicate the level of verbosity. By default, nothing is printed.
        """
        self.srm = srm
        self.n_pieces = n_pieces
        self.clustering = clustering
        self.n_bags = n_bags
        self.mask = mask
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.detrend = detrend
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.n_jobs = n_jobs
        self.memory = None
        self.memory_level = 0
        self.verbose = verbose
        self.reshape = reshape

    def fit(self, imgs):
        """
        Learn a template from source images, using alignment.

        Parameters
        ----------
        imgs: List of 4D Niimg-like or List of lists of 3D Niimg-like
            Source subjects data. Each element of the parent list is one subject
            data, and all must have the same length (n_samples).

        Returns
        -------
        self

        Attributes
        ----------
        self.template: 4D Niimg object
            Length : n_samples

        """
        # Check if the input is a list, if list of lists, concatenate each subjects
        # data into one unique image.
        if not isinstance(imgs, (list, np.ndarray)) or len(imgs) < 2:
            raise ValueError(
                "The method TemplateAlignment.fit() need a list as input. "
                "Each element of the list (Niimg-like or list of Niimgs) "
                "is the data for one subject."
            )
        else:
            if isinstance(imgs[0], (list, np.ndarray)):
                imgs = [concat_imgs(img) for img in imgs]

        self.masker_ = check_embedded_masker(self)
        self.masker_.n_jobs = self.n_jobs  # self.n_jobs

        # if masker_ has been provided a mask_img
        if self.masker_.mask_img is None:
            self.masker_.fit(imgs)
        else:
            self.masker_.fit()

        if isinstance(self.clustering, nib.nifti1.Nifti1Image) or os.path.isfile(
            self.clustering
        ):
            # check that clustering provided fills the mask, if not, reduce the mask
            if 0 in self.masker_.transform(self.clustering):
                reduced_mask = _intersect_clustering_mask(
                    self.clustering, self.masker_.mask_img
                )
                self.mask = reduced_mask
                self.masker_ = check_embedded_masker(self)
                self.masker_.n_jobs = self.n_jobs
                self.masker_.fit()
                warnings.warn(
                    "Mask used was bigger than clustering provided. "
                    + "Its intersection with the clustering was used instead."
                )

        rs = ShuffleSplit(n_splits=self.n_bags, test_size=0.8, random_state=0)

        outputs = Parallel(n_jobs=self.n_jobs, prefer="threads", verbose=self.verbose)(
            delayed(fit_one_parcellation)(
                imgs,
                self.srm,
                self.masker_,
                self.n_pieces,
                self.clustering,
                clustering_index,
                self.n_jobs,
                self.verbose,
            )
            for clustering_index, _ in rs.split(range(load_img(imgs[0]).shape[-1]))
        )

        self.labels_ = [output[0] for output in outputs]
        self.fit_ = [output[1] for output in outputs]
        self.reduced_sr = [output[2] for output in outputs]
        return self

    def add_subjects(self, imgs):
        """Add subject without recalculating SR"""
        for labels, srm, reduced_sr in zip(self.labels_, self.fit_, self.reduced_sr):
            for X_i, piece_srm, piece_sr in zip(
                list(generate_X_is(labels, imgs, self.masker_, self.verbose)),
                srm,
                reduced_sr,
            ):
                piece_srm.add_subjects(X_i, piece_sr)
        return self

    def transform(self, imgs):
        """
        Parameters
        ----------
        imgs : list of Niimgs or string (paths). Masked shape : n_voxels, n_timeframes

        Returns
        -------
        reshaped_aligned

        !!!Not implemented for n_bags>1
        """
        if self.n_bags > 1:
            warnings.warn("n_bags > 1 is not yet supported for this method.")

        n_comps = self.srm.n_components
        aligned_imgs = []
        for labels, srm in zip(self.labels_, self.fit_):
            bag_align = []
            for X_i, piece_srm in zip(
                list(generate_X_is(labels, imgs, self.masker_, self.verbose)),
                srm,
            ):
                piece_align = piece_srm.transform(X_i)
                p_comps = piece_srm.n_components
                if p_comps != n_comps:
                    piece_align = [
                        np.pad(
                            t,
                            ((0, n_comps - p_comps), (0, 0)),
                            mode="constant",
                        )
                        for t in piece_align
                    ]
                bag_align.append(piece_align)
            aligned_imgs.append(bag_align)
        reordered_aligned = np.moveaxis(aligned_imgs[0], [1], [0])
        if self.reshape is False:
            return reordered_aligned
        reshaped_aligned = reordered_aligned.reshape(
            len(imgs), -1, np.shape(aligned_imgs)[-1]
        )
        return reshaped_aligned

    def get_full_basis(self):
        """
        Concatenated (and padded if needed) local basis for each subject to
        fullbrain basis of shape (n_components,n_voxels) for each subject
        """
        unique_labels = np.unique(self.labels_[0])
        n_comps = self.srm.n_components
        n_subs = len(self.fit_[0][0].basis_list)
        full_basis_list = np.zeros(shape=(n_subs, len(self.labels_[0]), n_comps))

        for k in range(len(unique_labels)):
            label = unique_labels[k]
            k_estim = self.fit_[0][k]
            i = label == self.labels_[0]
            p_comps = k_estim.n_components
            for s, basis in enumerate(k_estim.basis_list):
                full_basis_list[s, i, :] = np.pad(
                    basis, ((0, n_comps - p_comps), (0, 0)), mode="constant"
                ).T

        basis_imgs = [
            self.masker_.inverse_transform(full_basis.T)
            for full_basis in full_basis_list
        ]
        return basis_imgs

    def fit_transform(self, imgs):
        return self.fit(imgs).transform(imgs)

    def inverse_transform(self, X_transformed):
        # TODO: Just inverse operations in transform to make API compliant
        pass

    def clean(self):
        """
        Just clean temporary directories
        """
        for srm in self.fit_:
            srm.clean()
