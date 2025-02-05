# -*- coding: utf-8 -*-
"""
Module for functional template inference.
Uses functional alignment on Niimgs and predicts new subjects' unseen images.
"""
# Author: T. Bazeille, B. Thirion
# License: simplified BSD

import numpy as np
from joblib import Memory, Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from fmralign._utils import _parcels_to_array, _transform_one_img
from fmralign.pairwise_alignment import PairwiseAlignment, fit_one_piece
from fmralign.preprocessing import ParcellationMasker


def _rescaled_euclidean_mean(subjects_data, scale_average=False):
    """
    Make the Euclidian average of `numpy.ndarray`.

    Parameters
    ----------
    subjects_data: `list` of `numpy.ndarray`
        Each element of the list is the data for one subject.
    scale_average: boolean
        If true, average is rescaled so that it keeps the same norm as the
        average of training images.

    Returns
    -------
    average_data: ndarray
        Average of imgs, with same shape as one img
    """
    average_data = np.mean(subjects_data, axis=0)
    scale = 1
    if scale_average:
        X_norm = 0
        for data in subjects_data:
            X_norm += np.linalg.norm(data)
        X_norm /= len(subjects_data)
        scale = X_norm / np.linalg.norm(average_data)
    average_data *= scale

    return average_data


def _reconstruct_template(fit, labels, masker):
    """
    Reconstruct template from fit output.

    Parameters
    ----------
    fit: list of list of numpy.ndarray
        Each element of the list is the list of parcels data for one subject.
    labels: numpy.ndarray
        Labels of the parcels.
    masker: instance of NiftiMasker or MultiNiftiMasker
        Masker to be used on the data.

    Returns
    -------
    template_img: 4D Niimg object
        Models the barycenter of input imgs
    template_history: list of 4D Niimgs
        List of the intermediate templates computed at the end of each iteration
    """
    template_parcels = [fit_i["template_data"] for fit_i in fit]
    template_data = _parcels_to_array(template_parcels, labels)
    template_img = masker.inverse_transform(template_data)

    n_iter = len(fit[0]["template_history"])
    template_history = []
    for i in range(n_iter):
        template_parcels = [fit_j["template_history"][i] for fit_j in fit]
        template_data = _parcels_to_array(template_parcels, labels)
        template_history.append(masker.inverse_transform(template_data))

    return template_img, template_history


def _align_images_to_template(
    subjects_data,
    template,
    alignment_method,
):
    """
    Convenience function.
    For a list of ndarrays, return the list of alignment estimators
    aligning each of them to a common target, the template.

    Parameters
    ----------
    subjects_data: `list` of `numpy.ndarray`
        Each element of the list is the data for one subject.
    template: `numpy.ndarray`
        The target data.
    alignment_method: string
        Algorithm used to perform alignment between sources and template.

    Returns
    -------
    aligned_data: `list` of `numpy.ndarray`
        List of aligned data.
    piecewise_estimators: `list` of `PairwiseAlignment`
        List of `Alignment` estimators.
    """
    aligned_data = []
    piecewise_estimators = []
    for subject_data in subjects_data:
        piecewise_estimator = fit_one_piece(
            subject_data,
            template,
            alignment_method,
        )
        piecewise_estimator.fit(subject_data, template)
        piecewise_estimators.append(piecewise_estimator)
        aligned_data.append(piecewise_estimator.transform(subject_data))
    return aligned_data, piecewise_estimators


def _fit_local_template(
    subjects_data,
    n_iter=2,
    scale_template=False,
    alignment_method="identity",
):
    """
    Create template through alternate minimization.
    Compute iteratively :
    * T minimizing sum(||R X-T||) which is the mean of aligned images (RX)
    * align initial images to new template T
        (find transform R minimizing ||R X-T|| for each img X)


    Parameters
    ----------
    imgs: `list` of `numpy.ndarray`
        Each element of the list is the data for one subject.
    n_iter: int
       Number of iterations in the alternate minimization. Each image is
       aligned n_iter times to the evolving template. If n_iter = 0,
       the template is simply the mean of the input images.
    scale_template: boolean
        If true, template is rescaled after each inference so that it keeps
        the same norm as the average of training images.
    alignment_method: string
        Algorithm used to perform alignment between sources and template.

    Returns
    -------
    template_data: `numpy.ndarray`
        Template data.
    template_history: `list` of `numpy.ndarray`
        List of the intermediate templates computed at the end of each iteration.
    piecewise_estimators: `list` of `PairwiseAlignment`
        List of `Alignment` estimators.
    """

    aligned_data = subjects_data
    template_history = []
    for iter in range(n_iter):
        template = _rescaled_euclidean_mean(aligned_data, scale_template)
        if 0 < iter < n_iter - 1:
            template_history.append(template)
        aligned_data, subjects_estimators = _align_images_to_template(
            subjects_data,
            template,
            alignment_method,
        )

    return {
        "template_data": template,
        "template_history": template_history,
        "estimators": subjects_estimators,
    }


def _index_by_parcel(subjects_parcels):
    """
    Index data by parcel.

    Parameters
    ----------
    subjects_parcels: list of list of numpy.ndarray
        Each element of the list is the list of parcels
        data for one subject.

    Returns
    -------
    list of list of numpy.ndarray
        Each element of the list is the list of subjects
        data for one parcel.
    """
    n_pieces = subjects_parcels[0].n_pieces
    return [
        [subject_parcels[i] for subject_parcels in subjects_parcels]
        for i in range(n_pieces)
    ]


class TemplateAlignment(BaseEstimator, TransformerMixin):
    """
    Decompose the source images into regions and summarize subjects information
    in a template, then use pairwise alignment to predict
    new contrast for target subject.
    """

    def __init__(
        self,
        alignment_method="identity",
        n_pieces=1,
        clustering="kmeans",
        scale_template=False,
        n_iter=2,
        save_template=None,
        mask=None,
        smoothing_fwhm=None,
        standardize=False,
        detrend=None,
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
        """
        Parameters
        ----------
        alignment_method: string
            Algorithm used to perform alignment between X_i and Y_i :
            * either 'identity', 'scaled_orthogonal', 'optimal_transport',
            'ridge_cv', 'diagonal',
            * or an instance of one of alignment classes (imported from
            fmralign.alignment_methods)
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
            n_pieces is then ignored.
        scale_template: boolean, default False
            rescale template after each inference so that it keeps
            the same norm as the average of training images.
        n_iter: int
           number of iteration in the alternate minimization. Each img is
           aligned n_iter times to the evolving template. If n_iter = 0,
           the template is simply the mean of the input images.
        save_template: None or string(optional)
            If not None, path to which the template will be saved.
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
        self.template = None
        self.template_history = None
        self.alignment_method = alignment_method
        self.n_pieces = n_pieces
        self.clustering = clustering
        self.n_iter = n_iter
        self.scale_template = scale_template
        self.save_template = save_template
        self.mask = mask
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.detrend = detrend
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose

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

        subjects_parcels = self.parcel_masker.fit_transform(imgs)
        parcels_data = _index_by_parcel(subjects_parcels)
        self.masker = self.parcel_masker.masker_
        self.mask = self.parcel_masker.masker_.mask_img_
        self.labels_ = self.parcel_masker.labels
        self.n_pieces = self.parcel_masker.n_pieces

        self.fit_ = Parallel(
            self.n_jobs, prefer="threads", verbose=self.verbose
        )(
            delayed(_fit_local_template)(
                parcel_i,
                self.n_iter,
                self.scale_template,
                self.alignment_method,
            )
            for parcel_i in parcels_data
        )

        self.template, self.template_history = _reconstruct_template(
            self.fit_, self.labels_, self.masker
        )
        if self.save_template is not None:
            self.template.to_filename(self.save_template)

    def transform(self, img, subject_index=None):
        """
        Transform a (new) subject image into the template space.

        Parameters
        ----------
        img: 4D Niimg-like object
            Subject image.
        subject_index: int, optional (default = None)
            Index of the subject to be transformed. It should
            correspond to the index of the subject in the list of
            subjects used to fit the template. If None, a new
            `PairwiseAlignment` object is fitted between the new
            subject and the template before transforming.


        Returns
        -------
        predicted_imgs: 4D Niimg object
            Transformed data.

        """
        if not hasattr(self, "fit_"):
            raise ValueError(
                "This instance has not been fitted yet. "
                "Please call 'fit' before 'transform'."
            )

        if subject_index is None:
            alignment_estimator = PairwiseAlignment(
                n_pieces=self.n_pieces,
                alignment_method=self.alignment_method,
                clustering=self.parcel_masker.get_parcellation_img(),
                mask=self.masker,
                smoothing_fwhm=self.smoothing_fwhm,
                standardize=self.standardize,
                detrend=self.detrend,
                target_affine=self.target_affine,
                target_shape=self.target_shape,
                low_pass=self.low_pass,
                high_pass=self.high_pass,
                t_r=self.t_r,
                memory=self.memory,
                memory_level=self.memory_level,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
            alignment_estimator.fit(img, self.template)
            return alignment_estimator.transform(img)
        else:
            parceled_data_list = self.parcel_masker.transform(img)
            subject_estimators = [
                fit_i["estimators"][subject_index] for fit_i in self.fit_
            ]
            transformed_img = Parallel(
                self.n_jobs, prefer="threads", verbose=self.verbose
            )(
                delayed(_transform_one_img)(parceled_data, subject_estimators)
                for parceled_data in parceled_data_list
            )
            if len(transformed_img) == 1:
                return transformed_img[0]
            else:
                return transformed_img

    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here. Will raise AttributeError if called."""
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
