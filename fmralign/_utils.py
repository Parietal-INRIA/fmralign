# -*- coding: utf-8 -*-
import warnings
from collections import defaultdict

import nibabel as nib
import numpy as np
import torch
from nilearn._utils.niimg_conversions import check_same_fov
from nilearn.image import concat_imgs, new_img_like, smooth_img
from nilearn.maskers import NiftiLabelsMasker, SurfaceLabelsMasker
from nilearn.maskers._utils import concatenate_surface_images
from nilearn.masking import apply_mask_fmri, intersect_masks
from nilearn.regions.parcellations import Parcellations
from nilearn.surface import SurfaceImage
from pathlib import Path
import joblib
import datetime
from sklearn.exceptions import NotFittedError


class ParceledData:
    """A class for managing parceled data, such as neuroimaging data.

    Parameters
    ----------
        data: 2D :obj:`numpy.ndarray`
            Signal for each :term:`voxel` inside the mask.
            shape: (number of scans, number of voxels)
        masker: nilearn.maskers.NiftiMasker
            The masker used to transform the data.
        labels: `list` of `int`
            The parcels labels.
    """

    def __init__(self, data, masker, labels):
        self.data = data
        self.masker = masker
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.n_pieces = len(self.unique_labels)

    def __getitem__(self, key):
        """Retrieve data for a specific parcel or a list of parcels.

        Parameters
        ----------
        key: int or slice
            The index or slice of the parcels to retrieve.

        Returns
        -------
        numpy.ndarray: 2D array of shape (n_samples, n_features)
            or `list` of 2D arrays is a slice is provided.
            The data for the specified parcel(s).
        """
        if isinstance(key, int):
            return self.data[:, self.labels == self.unique_labels[key]]
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.n_pieces
            step = key.step if key.step is not None else 1
            return [
                self.data[:, self.labels == self.unique_labels[i]]
                for i in range(start, stop, step)
            ]
        else:
            raise ValueError("Invalid key type.")

    def get_parcel(self, label):
        """Retrieve data for specific parcel labels.

        Parameters
        ----------
        label: int
            The label of the parcel to retrieve.

        Returns
        -------
        numpy.ndarray: 2D array of shape (n_samples, n_features)
            The data for the specified parcel.
        """
        return self.data[:, self.labels == label]

    def to_list(self):
        """Convert the parceled data to a list of numpy.ndarray.

        Returns
        -------
        list: A list of numpy.ndarray, where each element
            corresponds to the data for a parcel.
        """
        if isinstance(self.data, np.ndarray):
            return [self[i] for i in range(self.n_pieces)]

    def to_img(self):
        """Convert the parceled data back to an image.

        Returns
        -------
        nibabel.Nifti1Image: The image reconstructed from the parceled data.
        """
        return self.masker.inverse_transform(self.data)


def _transform_one_img(parceled_data, fit):
    """Apply a transformation to a single `ParceledData` object."""
    transformed_data = piecewise_transform(
        parceled_data,
        fit,
    )
    transformed_img = transformed_data.to_img()
    return transformed_img


def _img_to_parceled_data(img, masker, labels):
    """Convert a 3D Niimg to a ParceledData object."""
    data = masker.transform(img)
    return ParceledData(data, masker, labels)


def _parcels_to_array(parceled_img, labels):
    """Convert a list of parcels  to a 2D array."""
    unique_labels = np.unique(labels)
    n_features = len(labels)
    n_samples = parceled_img[0].shape[0]
    data = np.zeros((n_samples, n_features))
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        data[:, labels == label] = parceled_img[i]
    return data


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
    """Apply a piecewise transform to parceled_data.

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
    """Compute a parcellation of the data.

    Use nilearn Parcellation class in our pipeline. It is used to find local
    regions of the brain in which alignment will be later applied. For
    alignment computational efficiency, regions should be of hundreds of
    voxels.

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
    if isinstance(clustering, nib.nifti1.Nifti1Image):
        check_same_fov(masker.mask_img_, clustering)
        labels = apply_mask_fmri(clustering, masker.mask_img_).astype(int)

    elif isinstance(clustering, SurfaceImage):
        labels = (
            np.vstack(
                [
                    clustering.data.parts["left"],
                    clustering.data.parts["right"],
                ]
            )
            .astype(int)
            .ravel()
        )

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
        labels = masker.transform(parcellation.labels_img_)[0].astype(int)

    if verbose > 0:
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"The alignment will be applied on parcels of sizes {counts}")

    # raise warning if some parcels are bigger than 1000 voxels
    _check_labels(labels)

    return labels


def _sparse_cluster_matrix(arr):
    """
    Creates a sparse matrix where element (i,j) is 1 if arr[i] == arr[j], 0 otherwise.

    Parameters
    ----------
    arr: torch.Tensor of shape (n,)
        1D array of integers

    Returns
    -------
    sparse_matrix: sparse torch.Tensor of shape (len(arr), len(arr))
    """

    n = len(arr)

    # Create a dictionary mapping each value to its indices
    value_to_indices = defaultdict(list)
    for i, val in enumerate(arr.tolist()):
        value_to_indices[val].append(i)

    # Create lists to store indices and values for the sparse matrix
    rows = []
    cols = []

    # For each value, add all pairs of indices where that value appears
    for indices in value_to_indices.values():
        for i in indices:
            rows += [i] * len(indices)
            cols += indices

    # Convert to tensors
    rows = torch.tensor(rows)
    cols = torch.tensor(cols)
    values = torch.ones(len(rows), dtype=torch.bool)

    # Create sparse tensor
    sparse_matrix = torch.sparse_coo_tensor(
        indices=torch.stack([rows, cols]),
        values=values,
        size=(n, n),
    ).coalesce()

    return sparse_matrix


def save_alignment(alignment_estimator, output_path):
    """Save the alignment estimator object to a file.

    Parameters
    ----------
    alignment_estimator : :obj:`PairwiseAlignment` or :obj:`TemplateAlignment`
        The alignment estimator object to be saved.
        It should be an instance of either `PairwiseAlignment` or
        `TemplateAlignment`.
        The object should have been fitted before saving.
    output_path : str or Path
        Path to the file or directory where the model will be saved.
        If a directory is provided, the model will be saved with a
        timestamped filename in that directory.
        If a file is provided, the model will be saved with that filename.

    Raises
    ------
    NotFittedError
        If the alignment estimator has not been fitted yet.
    ValueError
        If the output path is not a valid file or directory.
    """
    if not hasattr(alignment_estimator, "fit_"):
        raise NotFittedError(
            "This instance has not been fitted yet. "
            "Please call 'fit' before 'save'."
        )

    output_path = Path(output_path)

    if output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        suffix = f"alignment_estimator_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        joblib.dump(alignment_estimator, output_path / suffix)

    else:
        joblib.dump(alignment_estimator, output_path)


def load_alignment(input_path):
    """Load an alignment estimator object from a file.

    Parameters
    ----------
    input_path : str or Path
        Path to the file or directory from which the model will be loaded.
        If a directory is provided, the latest .pkl file in that directory
        will be loaded.

    Returns
    -------
    alignment_estimator : :obj:`PairwiseAlignment` or :obj:`TemplateAlignment`
        The loaded alignment estimator object.
        It will be an instance of either `PairwiseAlignment` or
        `TemplateAlignment`, depending on what was saved.

    Raises
    ------
    ValueError
        If no .pkl files are found in the directory or if the input path is not
        a valid file or directory.
    """
    input_path = Path(input_path)

    if input_path.is_dir():
        # If it's a directory, look for the latest .pkl file
        pkl_files = list(input_path.glob("*.pkl"))
        if not pkl_files:
            raise ValueError(
                f"No .pkl files found in the directory: {input_path}"
            )
        input_path = max(pkl_files, key=lambda x: x.stat().st_mtime)

    return joblib.load(input_path)


def get_connectivity_features(img, parcelation_img, masker):
    """Compute connectivity features for a single subject.

    Parameters
    ----------
    img : Nifti1Image | SurfaceImage
        Input subject image.
    parcelation_img : Nifti1Image | SurfaceImage
        Labels image for connectivity seeds.
    masker : NiftiMasker | SurfaceMasker
        Masker used to transform the data.

    Returns
    -------
    Nifti1Image | SurfaceImage
        Connectivity features image.
    """

    # Generate a LabelsMasker with the same parameters as the original masker
    allowed_keys = {
        "smoothing_fwhm",
        "standardize",
        "standardize_confounds",
        "detrend",
        "low_pass",
        "high_pass",
        "t_r",
        "memory",
        "memory_level",
        "verbose",
    }
    params = {
        k: v for k, v in masker.get_params().items() if k in allowed_keys
    }
    if params["standardize"] is False:
        raise ValueError(
            "Standardization is required for connectivity features."
        )
    if isinstance(parcelation_img, SurfaceImage):
        connectivity_targets = SurfaceLabelsMasker(
            labels_img=parcelation_img, **params
        ).fit_transform(img)
    else:
        connectivity_targets = NiftiLabelsMasker(
            labels_img=parcelation_img, **params
        ).fit_transform(img)

    # Compute the correlation features (n_targets x n_voxels)
    data = masker.transform(img)
    correlation_features = (
        connectivity_targets.T @ data / connectivity_targets.shape[0]
    )
    return masker.inverse_transform(correlation_features)


def get_modality_features(imgs, parcellation_img, masker, modality="response"):
    """Compute alignment features for the given modality.

    Parameters
    ----------
    imgs : Iterable[Nifti1Image  |  SurfaceImage]
        Iterable of images to be aligned.
    parcelation_img : Nifti1Image | SurfaceImage
        Labels image for connectivity seeds.
    masker : NiftiMasker | SurfaceMasker
        Masker used to transform the data.
    modality : str, optional
        modality : str, optional (default='response')
        Specifies the alignment modality to be used:
        * 'response': Aligns by directly comparing corresponding similar
        time points in the source and target images.
        * 'connectivity': Aligns based on voxel-wise connectivity features
        within each parcel, comparing how each voxel relates to others in
        the same region.
        * 'hybrid': Combines both time series and connectivity information
        to perform the alignment.

    Returns
    -------
    Iterable[Nifti1Image]
        List of images with additional features for alignment based on the
        specified modality.
        If modality is 'response', the original images are returned.
        If modality is 'connectivity', the connectivity features are returned.
        If modality is 'hybrid', the original images and connectivity features
        are concatenated and returned.

    Raises
    ------
    ValueError
        If the modality is not one of 'response', 'connectivity', or 'hybrid'.
    """
    if modality == "response":
        return imgs

    elif modality == "connectivity":
        connectivity_imgs = []
        for img in imgs:
            connectivity_imgs.append(
                get_connectivity_features(img, parcellation_img, masker)
            )
        return connectivity_imgs

    elif modality == "hybrid":
        hybrid_imgs = []
        for img in imgs:
            connectivity_img = get_connectivity_features(
                img, parcellation_img, masker
            )
            if isinstance(img, SurfaceImage):
                hybrid_img = concatenate_surface_images(
                    [img, connectivity_img]
                )
            else:
                hybrid_img = concat_imgs([img, connectivity_img])
            hybrid_imgs.append(hybrid_img)
        return hybrid_imgs

    else:
        raise ValueError(
            "mode must be 'response', 'connectivity', or 'hybrid'."
        )
