import numpy as np
from scipy.stats import pearsonr
from nilearn.regions.parcellations import Parcellations
from nilearn.image import smooth_img
from nilearn.masking import _apply_mask_fmri
from nilearn._utils.niimg_conversions import _check_same_fov
import nibabel


def piecewise_transform(labels, estimators, X):
    """ Apply a piecewise transform to X:
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

    X_transform = np.zeros_like(X)
    # Labels are from 1 to n where as estimators are indexed from 0 to n-1
    for i in np.unique(labels):
        X_transform[:, labels == i] = estimators[i - 1].transform(
            X[:, labels == i])
    return X_transform


def _make_parcellation(imgs, clustering, n_pieces, masker, smoothing_fwhm=5, verbose=0):
    """Convenience function to use nilearn Parcellation class in our pipeline.
    It is used to find local regions of the brain in which alignment will be later applied.
    For alignment computational efficiency, regions should be of hundreds of voxels.

    Parameters
    ----------
    imgs: Niimgs
        data to cluster
    clustering: string or 3D Niimg
        In : {'kmeans', 'ward', 'rena'}, passed to nilearn Parcellations class.
        If you aim for speed, choose k-means (and check kmeans_smoothing_fwhm parameter)
        If you want spatially connected and/or reproducible regions use 'ward'
        For 'rena', need nilearn > 0.5.2
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
    if type(clustering) == nibabel.nifti1.Nifti1Image:
        # check image makes suitable labels, this will return friendly error message if needed
        _check_same_fov(masker.mask_img_, clustering)
        labels_img = clustering
    else:
        if clustering == "kmeans" and smoothing_fwhm is not None:
            images_to_parcel = smooth_img(imgs, smoothing_fwhm)
        try:
            parcellation = Parcellations(method=clustering, n_parcels=n_pieces, mask=masker,
                                         scaling=False, n_iter=20, verbose=verbose)
            parcellation.fit()
        except TypeError:
            if clustering == "rena":
                raise InputError(
                    ('ReNA algorithm is only available in Nilearn version > 0.5.2. \
                     If you want to use it, please run "pip install --upgrade nilearn"'))
            else:
                parcellation = Parcellations(
                    method=clustering, n_parcels=n_pieces, mask=masker, verbose=verbose)
        parcellation.fit(imgs)
        labels_img = parcellation.labels_img_
    return _apply_mask_fmri(labels_img, masker.mask_img_).astype(int)


def voxelwise_correlation(ground_truth, prediction, masker):
    """
    Parameters
    ----------
    ground_truth: 3D or 4D Niimg
        Reference image (data acquired but never used before and considered as missing)
    prediction : 3D or 4D Niimg
        Same shape as ground_truth
    masker: instance of NiftiMasker or MultiNiftiMasker
        Masker to be used on ground_truth and prediction. For more information see:
        http://nilearn.github.io/manipulating_images/masker_objects.html

    Returns
    -------
    voxelwise_correlation : 3D Niimg
        Voxelwise score between ground_truth and prediction
    """
    X_gt = masker.transform(ground_truth)
    X_pred = masker.transform(prediction)

    voxelwise_correlation = np.array([pearsonr(X_gt[:, vox], X_pred[:, vox])[0]
                                      for vox in range(X_pred.shape[1])])
    return masker.inverse_transform(voxelwise_correlation)
