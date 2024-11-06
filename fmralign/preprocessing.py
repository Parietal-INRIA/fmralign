import nibabel as nib
import warnings
import os
import numpy as np

from typing import Sequence, Any

from nilearn._utils.masker_validation import check_embedded_masker
from nilearn.image import concat_imgs
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Memory, Parallel, delayed

from fmralign._utils import (
    _intersect_clustering_mask,
    _make_parcellation,
)


def _transform_img(img, masker, labels) -> Sequence[Any]:
    if img.shape[:-1] != masker.mask_img_.shape:
        raise ValueError(
            f"All images must have the same shape as the mask. "
            f"Expected shape {masker.mask_img_.shape}, "
            f"but got {img.shape[:-1]}."
        )
    data = masker.transform(img)
    X_ = []
    for i in np.unique(labels):
        X_.append(data[:, labels == i])
    return X_


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
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
        self.n_pieces = n_pieces
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
        self.labels = None

    def _fit_masker(self, imgs: Sequence[Any]) -> None:
        self.masker_ = check_embedded_masker(self)
        self.masker_.n_jobs = self.n_jobs

        if isinstance(imgs, nib.nifti1.Nifti1Image):
            imgs = [imgs]
        # Assert that all images have the same shape
        if len(set([img.shape for img in imgs])) > 1:
            raise NotImplementedError(
                "fmralign does not support images of different shapes."
            )
        if self.masker_.mask_img is None:
            self.masker_.fit(imgs)
        else:
            self.masker_.fit()

        if isinstance(
            self.clustering, nib.nifti1.Nifti1Image
        ) or os.path.isfile(self.clustering):
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

    def _one_parcellation(self, imgs: Sequence[Any]) -> None:
        if isinstance(imgs, list):
            imgs = concat_imgs(imgs)
        self.labels = _make_parcellation(
            imgs,
            self.clustering,
            self.n_pieces,
            self.masker_,
            smoothing_fwhm=self.smoothing_fwhm,
            verbose=self.verbose,
        )

    def fit(
        self,
        imgs: Sequence[Any],
    ):
        self._fit_masker(imgs)
        self._one_parcellation(imgs)
        return self

    def transform(
        self,
        imgs: Sequence[Any],
    ) -> Sequence[Any]:
        if isinstance(imgs, nib.nifti1.Nifti1Image):
            imgs = [imgs]

        parcelled_data = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_img)(
                img,
                self.masker_,
                self.labels,
            )
            for img in imgs
        )
        return parcelled_data
