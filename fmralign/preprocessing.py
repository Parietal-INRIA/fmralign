import os
import warnings

import numpy as np
from joblib import Memory, Parallel, delayed
from nibabel.nifti1 import Nifti1Image
from nilearn._utils.masker_validation import check_embedded_masker
from nilearn.image import concat_imgs
from sklearn.base import BaseEstimator, TransformerMixin

from fmralign._utils import _intersect_clustering_mask, _make_parcellation


def _transform_img(img, masker, labels):
    data = masker.transform(img)
    return ParcelledData(data, masker, labels)


class ParcelledData:
    def __init__(self, data, masker, labels):
        self.data = data
        self.masker = masker
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.n_pieces = len(self.unique_labels)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[:, self.labels == self.unique_labels[key]]
        elif isinstance(key, slice):
            raise NotImplementedError("Slicing is not implemented.")
        else:
            raise ValueError("Invalid key type.")

    def tolist(self):
        if isinstance(self.data, np.ndarray):
            return [self[i] for i in range(self.n_pieces)]

    def tonifti(self):
        return self.masker.inverse_transform(self.data)


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

    def _fit_masker(self, imgs):
        self.masker_ = check_embedded_masker(self)
        self.masker_.n_jobs = self.n_jobs

        if isinstance(imgs, Nifti1Image):
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

        if isinstance(self.clustering, Nifti1Image) or os.path.isfile(
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

    def _one_parcellation(self, imgs):
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

    def get_labels(self):
        if self.labels is None:
            raise ValueError(
                "Labels have not been computed yet,"
                "call fit before get_labels."
            )
        return self.labels

    def fit(self, imgs):
        self._fit_masker(imgs)
        self._one_parcellation(imgs)
        return self

    def transform(self, imgs):
        if isinstance(imgs, Nifti1Image):
            imgs = [imgs]

        parcelled_data = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_img)(
                img,
                self.masker_,
                self.labels,
            )
            for img in imgs
        )

        # Unpack the list if only one image was provided
        if len(imgs) == 1:
            return parcelled_data[0]
        else:
            return parcelled_data
