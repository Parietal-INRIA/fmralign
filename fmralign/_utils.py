# -*- coding: utf-8 -*-
import os
import warnings

import nibabel as nib
import numpy as np
from nilearn._utils.niimg_conversions import _check_same_fov
from nilearn.image import index_img, new_img_like, smooth_img
from nilearn.masking import _apply_mask_fmri, intersect_masks
from nilearn.regions.parcellations import Parcellations


def _intersect_clustering_mask(clustering, mask):
    """Take 3D Niimg clustering and bigger mask, output reduced mask."""
    dat = clustering.get_fdata()
    new_ = np.zeros_like(dat)
    new_[dat > 0] = 1
    clustering_mask = new_img_like(clustering, new_)
    return intersect_masks([clustering_mask, mask], threshold=1, connected=True)


def piecewise_transform(labels, estimators, X):
    """
    Apply a piecewise transform to X.

    Parameters
    ----------
    labels: list of ints (len n_features)
        Parcellation of features in clusters
    estimators: list of estimators with transform() method
        I-th estimator will be applied on the i-th cluster of features
    X: nd array (n_samples, n_features)
        Data to transform

    Returns
    -------
    X_transform: nd array (n_features, n_samples)
        Transformed data
    """
    unique_labels = np.unique(labels)
    X_transform = np.zeros_like(X)

    for i in range(len(unique_labels)):
        label = unique_labels[i]
        X_transform[:, labels == label] = estimators[i].transform(X[:, labels == label])
    return X_transform


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
    imgs, clustering_index, clustering, n_pieces, masker, smoothing_fwhm=5, verbose=0
):
    """
    Use nilearn Parcellation class in our pipeline.
    It is used to find local regions of the brain in which alignment will be later applied.
    For alignment computational efficiency, regions should be of hundreds of voxels.

    Parameters
    ----------
    imgs: Niimgs
        data to cluster
    clustering_index: list of integers
        Clustering is performed on a subset of the data chosen randomly
        in timeframes. This index carries this subset.
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
    if isinstance(clustering, nib.nifti1.Nifti1Image) or os.path.isfile(clustering):
        _check_same_fov(masker.mask_img_, clustering)
        labels = _apply_mask_fmri(clustering, masker.mask_img_).astype(int)

    # otherwise check it's needed, if not return 1 everywhere
    elif n_pieces == 1:
        labels = np.ones(int(masker.mask_img_.get_fdata().sum()), dtype=np.int8)

    # otherwise check requested clustering method
    elif isinstance(clustering, str) and n_pieces > 1:
        imgs_subset = index_img(imgs, clustering_index)
        if (clustering in ["kmeans", "hierarchical_kmeans"]) and (
            smoothing_fwhm is not None
        ):
            images_to_parcel = smooth_img(imgs_subset, smoothing_fwhm)
        else:
            images_to_parcel = imgs_subset
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
        labels = _apply_mask_fmri(parcellation.labels_img_, masker.mask_img_).astype(
            int
        )

    if verbose > 0:
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"The alignment will be applied on parcels of sizes {counts}")

    # raise warning if some parcels are bigger than 1000 voxels
    _check_labels(labels)

    return labels
