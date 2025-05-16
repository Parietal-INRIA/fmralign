"""Module for preprocessing data before alignment."""

import warnings

import numpy as np
from joblib import Memory, Parallel, delayed
from nibabel.nifti1 import Nifti1Image
from nilearn._utils.masker_validation import check_embedded_masker
from nilearn.image import concat_imgs
from nilearn.maskers._utils import concatenate_surface_images
from nilearn.surface import SurfaceImage
from sklearn.base import BaseEstimator, TransformerMixin

from fmralign._utils import (
    _img_to_parceled_data,
    _intersect_clustering_mask,
    _make_parcellation,
)


def _check_imgs(imgs):
    """Check if the input images are valid and convert them to a list."""
    if isinstance(imgs, (Nifti1Image, SurfaceImage)):
        imgs = [imgs]
    # If images are 3D, add a fourth dimension
    for i, img in enumerate(imgs):
        if len(img.shape) == 3:
            imgs[i] = Nifti1Image(
                np.expand_dims(img.get_fdata(), axis=-1),
                img.affine,
                img.header,
            )
    # Assert that all images have the same shape
    if len(set([img.shape for img in imgs])) > 1:
        raise NotImplementedError(
            "fmralign does not support images of different shapes."
        )
    return imgs


def _fit_new_masker(imgs, masker, n_jobs=1):
    """Fit a new masker on a single or multiple images."""
    masker.n_jobs = n_jobs
    return masker.fit(imgs)


class ParcellationMasker(BaseEstimator, TransformerMixin):
    """Class for masking Niimg-like objects and computing \
        a parcellation in a parallel fashion.

    Parameters
    ----------
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
    masker : None or :class:`~nilearn.surface.SurfaceImage`,\
           or :class:`nilearn.maskers.NiftiMasker`,\
           :class:`nilearn.maskers.MultiNiftiMasker` or \
           :class:`nilearn.maskers.SurfaceMasker`, optional
        Mask/Masker used for masking the data.
        If mask image if provided, it will be used in the MultiNiftiMasker or
        SurfaceMasker (depending on the type of mask image).
        If an instance of either maskers is provided, then this instance
        parameters will be used in masking the data by overriding the default
        masker parameters.
        If None, mask will be automatically computed by a MultiNiftiMasker
        with default parameters for Nifti images and no mask will be used for
        SurfaceImage.
    mask: Niimg-like object, optional (default = None)
        Mask to be used on data.
    smoothing_fwhm: float, optional (default = None)
        If smoothing_fwhm is not None, it gives the size in millimeters
        of the spatial smoothing to apply to the signal.
    standardize: boolean, optional (default = "zscore_sample")
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
    labels: list of ints, optional (default = None)
        Labels associated with the parcellation.
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

    def __init__(
        self,
        n_pieces=1,
        clustering="kmeans",
        masker=None,
        mask=None,
        smoothing_fwhm=None,
        standardize="zscore_sample",
        detrend=False,
        target_affine=None,
        target_shape=None,
        low_pass=None,
        high_pass=None,
        t_r=None,
        labels=None,
        memory=Memory(location=None),
        memory_level=0,
        n_jobs=1,
        verbose=0,
    ):
        self.n_pieces = n_pieces
        self.clustering = clustering
        self.masker = masker
        self.mask = mask
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.labels = labels
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _fit_masker(self, imgs):
        """Fit the masker on a single or multiple images."""
        if self.masker is None:
            masker_type = (
                "surface" if isinstance(imgs[0], SurfaceImage) else "multi_nii"
            )
            self.masker = check_embedded_masker(self, masker_type=masker_type)

        # Check if the masker has been fitted
        if not hasattr(self.masker, "mask_img_"):
            self.masker = _fit_new_masker(
                imgs, self.masker, n_jobs=self.n_jobs
            )

        # Check that the mask is not bigger than the clustering img
        if isinstance(self.clustering, Nifti1Image):
            # check that clustering provided fills the mask, if not, reduce the mask
            if 0 in self.masker.transform(self.clustering):
                reduced_mask = _intersect_clustering_mask(
                    self.clustering, self.masker.mask_img_
                )
                self.mask = reduced_mask
                self.masker = check_embedded_masker(self)
                self.masker.n_jobs = self.n_jobs
                self.masker.fit()
                warnings.warn(
                    "Mask used was bigger than clustering provided. "
                    + "Its intersection with the clustering was used instead."
                )

    def _one_parcellation(self, imgs):
        """Compute one parcellation for all images."""
        if isinstance(imgs, list):
            if isinstance(imgs[0], (Nifti1Image)):
                imgs = concat_imgs(imgs)
            else:
                imgs = concatenate_surface_images(imgs)
        self.labels = _make_parcellation(
            imgs,
            self.clustering,
            self.n_pieces,
            self.masker,
            smoothing_fwhm=self.smoothing_fwhm,
            verbose=self.verbose,
        )
        # Update the number of pieces if the
        # user provided a custom clustering
        self.n_pieces = len(np.unique(self.labels))

    def get_labels(self):
        """Return the labels associated with the parcellation.

        Returns
        -------
        list of ints (len n_features)
            The labels associated with the parcellation.

        Raises
        ------
        ValueError
            If the `.fit` method has not been called before.
        """
        if self.labels is None:
            raise ValueError(
                "Labels have not been computed yet,call fit before get_labels."
            )
        return self.labels

    def get_parcellation_img(self):
        """Return the parcellation image.

        Returns
        -------
        parcellation : `nibabel.Nifti1Image`
            Parcellation image.
        """
        return self.masker.inverse_transform(self.get_labels())

    def fit(self, imgs, y=None):
        """Fit the masker and compute the parcellation.

        Parameters
        ----------
        imgs : Niimg-like object or `list` of Niimg-like objects
            Images used to compute to fit the masker and compute the parcellation.
        y : None
            This parameter is unused. It is solely included for
            scikit-learn compatibility.

        """
        imgs = _check_imgs(imgs)
        self._fit_masker(imgs)
        self._one_parcellation(imgs)
        return self

    def transform(self, imgs):
        """Prepare data in parallel.

        Parameters
        ----------
        imgs : Niimg-like object or `list` of Niimg-like objects
            Images to be processed.

        Returns
        -------
        parcelled_data : `list` of ParceledData
            List of ParceledData objects containing the data and parcelation
            information for each image.
        """
        if isinstance(imgs, (Nifti1Image, SurfaceImage)):
            imgs = [imgs]

        parceled_data = Parallel(n_jobs=self.n_jobs)(
            delayed(_img_to_parceled_data)(
                img,
                self.masker,
                self.labels,
            )
            for img in imgs
        )

        return parceled_data
