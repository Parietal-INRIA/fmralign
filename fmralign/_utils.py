# -*- coding: utf-8 -*-
import os
import warnings

import nibabel as nib
import numpy as np
from nilearn._utils.niimg_conversions import check_same_fov
from nilearn.image import new_img_like, smooth_img
from nilearn.masking import apply_mask_fmri, intersect_masks
from nilearn.regions.parcellations import Parcellations


class ParceledData:
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


def transform_one_img(parceled_data, fit):
    transformed_data = piecewise_transform(
        parceled_data,
        fit,
    )
    transformed_img = transformed_data.tonifti()
    return transformed_img


def _img_to_parceled_data(img, masker, labels):
    data = masker.transform(img)
    return ParceledData(data, masker, labels)


def _parcels_to_array(parceled_img, labels):
    unique_labels = np.unique(labels)
    n_features = len(labels)
    n_samples = parceled_img[0].shape[0]
    aggregate_img = np.zeros((n_samples, n_features))
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        aggregate_img[:, labels == label] = parceled_img[i]
    return aggregate_img


def _intersect_clustering_mask(clustering, mask):
    """Take 3D Niimg clustering and bigger mask, output reduced mask."""
    dat = clustering.get_fdata()
    new_ = np.zeros_like(dat)
    new_[dat > 0] = 1
    clustering_mask = new_img_like(clustering, new_)
    return intersect_masks(
        [clustering_mask, mask], threshold=1, connected=True
    )


def piecewise_transform(parceled_data, estimators):
    """
    Apply a piecewise transform to parceled_data.

    Parameters
    ----------
    parceled_data: ParceledData
        Data to transform
    estimators: list of estimators with transform() method
        I-th estimator will be applied on the i-th cluster of features

    Returns
    -------
    parceled_data: ParceledData
        Transformed data
    """
    transformed_data_list = []
    for i in range(len(estimators)):
        transformed_data_list.append(estimators[i].transform(parceled_data[i]))
    # Convert transformed_data_list to ParceledData
    parceled_data = ParceledData(
        _parcels_to_array(transformed_data_list, parceled_data.labels),
        parceled_data.masker,
        parceled_data.labels,
    )
    return parceled_data


def _remove_empty_labels(labels):
    """Remove empty values label values from labels list."""
    vals = np.unique(labels)
    inverse_vals = -np.ones(labels.max() + 1).astype(int)
    inverse_vals[vals] = np.arange(len(vals))
    return inverse_vals[labels]


def _check_labels(labels, threshold=1000):
    """Check if any parcels are bigger than set threshold."""
    unique_labels, counts = np.unique(labels, return_counts=True)

    if not all(count < threshold for count in counts):
        warning = (
            "\n Some parcels are more than 1000 voxels wide it can slow down alignment,"
            "especially optimal_transport :"
        )
        for i in range(len(counts)):
            if counts[i] > threshold:
                warning += f"\n parcel {unique_labels[i]} : {counts[i]} voxels"
        warnings.warn(warning)
    pass


def _make_parcellation(
    imgs, clustering, n_pieces, masker, smoothing_fwhm=5, verbose=0
):
    """
    Use nilearn Parcellation class in our pipeline.
    It is used to find local regions of the brain in which alignment will be later applied.
    For alignment computational efficiency, regions should be of hundreds of voxels.

    Parameters
    ----------
    imgs: Niimgs
        data to cluster
    clustering: string or 3D Niimg
        If you aim for speed, choose k-means (and check kmeans_smoothing_fwhm parameter)
        If you want spatially connected and/or reproducible regions use 'ward'
        If you want balanced clusters (especially from timeseries) used 'hierarchical_kmeans'
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
    # check if clustering is provided
    if isinstance(clustering, nib.nifti1.Nifti1Image) or os.path.isfile(
        clustering
    ):
        check_same_fov(masker.mask_img_, clustering)
        labels = apply_mask_fmri(clustering, masker.mask_img_).astype(int)

    # otherwise check it's needed, if not return 1 everywhere
    elif n_pieces == 1:
        labels = np.ones(
            int(masker.mask_img_.get_fdata().sum()), dtype=np.int8
        )

    # otherwise check requested clustering method
    elif isinstance(clustering, str) and n_pieces > 1:
        if (clustering in ["kmeans", "hierarchical_kmeans"]) and (
            smoothing_fwhm is not None
        ):
            images_to_parcel = smooth_img(imgs, smoothing_fwhm)
        else:
            images_to_parcel = imgs
        parcellation = Parcellations(
            method=clustering,
            n_parcels=n_pieces,
            mask=masker,
            scaling=False,
            n_iter=20,
            verbose=verbose,
        )
        try:
            parcellation.fit(images_to_parcel)
        except ValueError as err:
            errmsg = (
                f"Clustering method {clustering} should be supported by "
                "nilearn.regions.Parcellation or a 3D Niimg."
            )
            err.args += (errmsg,)
            raise err
        labels = apply_mask_fmri(
            parcellation.labels_img_, masker.mask_img_
        ).astype(int)

    if verbose > 0:
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"The alignment will be applied on parcels of sizes {counts}")

    # raise warning if some parcels are bigger than 1000 voxels
    _check_labels(labels)

    return labels
