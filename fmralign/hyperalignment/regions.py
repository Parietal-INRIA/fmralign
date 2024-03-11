"""Utilities for computing searchlights. Adapted from ```nilearn```.\n
See the ```nilearn``` documentation for more details:
- https://nilearn.github.io/modules/generated/nilearn.regions.Parcellations.html
- https://nilearn.github.io/dev/modules/generated/nilearn.decoding.SearchLight.html

Author: Denis Fouchard, INRIA Saclay, MIND, 2023.
"""

import functools
import numpy as np
from joblib import Parallel, delayed
from nilearn import image, masking
from nilearn._utils import check_niimg_4d, check_niimg_3d
from nilearn.image.resampling import coord_transform
import warnings
from sklearn import neighbors
from scipy.spatial import distance_matrix
from nilearn._utils.niimg_conversions import (
    safe_get_data,
)
from .linalg import procrustes
from .linalg import ridge
from .local_template import compute_template
from fmralign._utils import _make_parcellation
from nilearn.maskers import NiftiMasker
from nibabel.nifti1 import Nifti1Image

###################################################################################################
# Compute parcels
###################################################################################################


def create_parcels_from_labels(labels: np.ndarray):
    """

    Parameters:
    ----------
    labels : ndarray
        Array of labels.

    Returns:
    -------
    parcles : list
        List of parcels, where each parcel is an array of indices.
    """
    n_labels = labels.max()
    parcels = []
    for i in range(1, n_labels + 1):
        parcels.append(np.where(labels == i)[0])
    return parcels


def compute_parcels(
    niimg, mask, n_parcels=100, verbose=True, smoothing_fwhm=5, n_jobs=1
):
    """
    Compute parcels using a given mask and input image.

    Parameters:
    - niimg: Input image to be parcellated.
    - mask: Mask image defining the region of interest.
    - n_parcels: Number of parcels to be created (default: 100).
    - verbose: Whether to print progress messages (default: True).
    - smoothing_fwhm: Full Width at Half Maximum (FWHM) for smoothing (default: 5).
    - n_jobs: Number of parallel jobs to run (default: 1).

    Returns:
    - Parcels created from the input image and mask.
    """
    if verbose:
        print("[Loading/Parcel] : Parcellating...")

    if isinstance(mask, Nifti1Image):
        mask = NiftiMasker(
            mask_img=mask, standardize=True, smoothing_fwhm=smoothing_fwhm
        )
    # Parcellation
    indexes = [1]
    labels = _make_parcellation(
        imgs=niimg,
        clustering_index=indexes,
        clustering="kmeans",
        n_pieces=n_parcels,
        masker=mask,
        smoothing_fwhm=mask.smoothing_fwhm,
    )

    parcels = create_parcels_from_labels(labels)

    print("Minimum length parcel: ", min([len(p) for p in parcels]))

    return parcels


###################################################################################################
# Computing searchlights
###################################################################################################


def _apply_mask_and_get_affinity(
    seeds, niimg, radius, allow_overlap, mask_img=None, n_jobs=1
):
    """Get only the rows which are occupied by sphere \
    at given seed locations and the provided radius.

    Rows are in target_affine and target_shape space.

    Parameters
    ----------
    seeds : List of triplets of coordinates in native space
        Seed definitions. List of coordinates of the seeds in the same space
        as target_affine.

    niimg : 3D/4D Niimg-like object
        See :ref:`extracting_data`.
        Images to process.
        If a 3D niimg is provided, a singleton dimension will be added to
        the output to represent the single scan in the niimg.

    radius : float
        Indicates, in millimeters, the radius for the sphere around the seed.

    allow_overlap : boolean
        If False, a ValueError is raised if VOIs overlap

    mask_img : Niimg-like object, optional
        Mask to apply to regions before extracting signals. If niimg is None,
        mask_img is used as a reference space in which the spheres 'indices are
        placed.

    Returns
    -------
    X : 2D numpy.ndarray
        Signal for each brain voxel in the (masked) niimgs.
        shape: (number of scans, number of voxels)

    A : scipy.sparse.lil_matrix
        Contains the boolean indices for each sphere.
        shape: (number of seeds, number of voxels)

    """
    seeds = list(seeds)

    # Compute world coordinates of all in-mask voxels.
    if niimg is None:
        mask, affine = masking._load_mask_img(mask_img)
        # Get coordinate for all voxels inside of mask
        mask_coords = np.asarray(np.nonzero(mask)).T.tolist()
        X = None

    elif mask_img is not None:
        affine = niimg.affine
        mask_img = check_niimg_3d(mask_img)
        mask_img = image.resample_img(
            mask_img,
            target_affine=affine,
            target_shape=niimg.shape[:3],
            interpolation="nearest",
        )
        mask, _ = masking.load_mask_img(mask_img)
        mask_coords = list(zip(*np.where(mask != 0)))

        X = masking.apply_mask_fmri(niimg, mask_img)

    elif niimg is not None:
        affine = niimg.affine
        if np.isnan(np.sum(safe_get_data(niimg))):
            warnings.warn(
                "The imgs you have fed into fit_transform() contains NaN "
                "values which will be converted to zeroes."
            )
            X = safe_get_data(niimg, True).reshape([-1, niimg.shape[3]]).T
        else:
            X = safe_get_data(niimg).reshape([-1, niimg.shape[3]]).T

        mask_coords = list(np.ndindex(niimg.shape[:3]))

    else:
        raise ValueError("Either a niimg or a mask_img must be provided.")

    # For each seed, get coordinates of nearest voxel
    nearests = []
    for sx, sy, sz in seeds:
        nearest = np.round(
            image.resampling.coord_transform(sx, sy, sz, np.linalg.inv(affine))
        )
        nearest = nearest.astype(int)
        nearest = (nearest[0], nearest[1], nearest[2])
        try:
            nearests.append(mask_coords.index(nearest))
        except ValueError:
            nearests.append(None)

    mask_coords = np.asarray(list(zip(*mask_coords)))
    mask_coords = image.resampling.coord_transform(
        mask_coords[0], mask_coords[1], mask_coords[2], affine
    )
    mask_coords = np.asarray(mask_coords).T

    clf = neighbors.NearestNeighbors(radius=radius, n_jobs=n_jobs)
    A = clf.fit(mask_coords).radius_neighbors_graph(seeds)
    A = A.tolil()
    for i, nearest in enumerate(nearests):
        if nearest is None:
            continue

        A[i, nearest] = True

    mask_coords_floats = mask_coords.copy()

    # Include the voxel containing the seed itself if not masked
    mask_coords = mask_coords.astype(int).tolist()
    for i, seed in enumerate(seeds):
        try:
            A[i, mask_coords.index(list(map(int, seed)))] = True
        except ValueError:
            # seed is not in the mask
            pass

    sphere_sizes = np.asarray(A.tocsr().sum(axis=1)).ravel()
    empty_spheres = np.nonzero(sphere_sizes == 0)[0]
    if len(empty_spheres) != 0:
        raise ValueError(f"These spheres are empty: {empty_spheres}")

    if (not allow_overlap) and np.any(A.sum(axis=0) >= 2):
        raise ValueError("Overlap detected between spheres")

    return X, A, mask_coords_floats


def compute_searchlights(
    niimg,
    mask_img,
    process_mask_img=None,
    radius=20,
    return_dist_mat=False,
    n_jobs=1,
):
    """
    Compute searchlights for a given 4D image and mask.

    Parameters
    ----------
    miimg : Niimg-like object
        See :ref:`extracting_data`.
        4D image.

    mask_img : Niimg-like object
        See :ref:`extracting_data`.
        Boolean image giving location of voxels containing usable signals.

    process_mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Boolean image giving voxels on which searchlight should be
        computed.

    radius : float, optional
        radius of the searchlight ball, in millimeters. Defaults to 20.

    return_dist_mat : bool, optional
        Whether to return the distance matrix between voxels in the mask.
        Defaults to False.

    verbose : int, optional
        Verbosity level (0 means no message).
        Defaults to 0.

    Returns
    -------
    X : 2D numpy.ndarray
        Signal for each brain voxel in the (masked) niimgs.
        shape: (number of scans, number of voxels)

    A_list : list of lists
        Contains the boolean indices for each sphere.
        shape: (number of seeds, number of voxels)

    dists : list of lists
        Contains the distance between each voxel and the seed.
        shape: (number of seeds, number of voxels)

    """

    # check if image is 4D
    niimg = check_niimg_4d(niimg)

    # Get the seeds
    if process_mask_img is None:
        process_mask_img = mask_img

    # Compute world coordinates of the seeds
    process_mask, process_mask_affine = masking.load_mask_img(process_mask_img)
    process_mask_coords = np.where(process_mask != 0)
    process_mask_coords = coord_transform(
        process_mask_coords[0],
        process_mask_coords[1],
        process_mask_coords[2],
        process_mask_affine,
    )
    process_mask_coords = np.asarray(process_mask_coords).T

    X, A, mask_coords = _apply_mask_and_get_affinity(
        process_mask_coords,
        niimg,
        radius=radius,
        allow_overlap=True,
        mask_img=mask_img,
        n_jobs=n_jobs,
    )

    A_list = []
    for i in range(A.shape[0]):
        A_list.append(A[i].nonzero()[1].tolist())

    dist_matrix = distance_matrix(mask_coords, mask_coords)
    dists = []
    for i, sl in enumerate(A_list):
        dists.append(dist_matrix[i, sl])

    if return_dist_mat:
        return X, A_list, dist_matrix

    return X, A_list, dists


def searchlight_weights(searchlights, dists, radius):
    """
    Calculate the weights for each searchlight based on the distances from the center.

    Parameters:
    ----------
    searchlights :list of arrays
        List of searchlights, where each searchlight is represented as an array of voxel indices.
    dists : array
        Array of distances from the center for each searchlight.
    radius : float
        Radius of the searchlight.

    Returns:
    --------
    weights : list
        List of weights for each searchlight.

    """
    nv = np.concatenate(searchlights).max() + 1
    weights_sum = np.zeros((nv,))
    for sl, d in zip(searchlights, dists):
        w = (radius - d) / radius
        weights_sum[sl] += w
    # print(np.percentile(weights_sum, np.linspace(0, 100, 11)))
    weights = []
    for sl, d in zip(searchlights, dists):
        w = (radius - d) / radius
        w /= weights_sum[sl]
        weights.append(w)
    return weights


###################################################################################################
# Hyperalignment
###################################################################################################


def iter_hyperalignment(
    X,
    Y,
    regions,
    sl_func,
    return_betas=False,
):
    """
    Tool function to iterate hyperalignment over pieces of data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The source data matrix.
    Y : array-like of shape (n_samples, n_features)
        The target data matrix.
    regions : array-like
        The indices of the regions.
    sl_func : function
        The function to use for hyperalignment.
    weights : array-like, optional
        The weights to use for weighted hyperalignment. Defaults to None.
    return_betas : bool, optional
        Whether to return the coefficients of regression instead of the prediciton.
        Defaults to False.

    Returns
    -------
    res : array-like
        The transformed data matrix.

    """
    if return_betas:
        T = np.zeros((X.shape[1], Y.shape[1]), dtype=np.float32)
    else:
        Yhat = np.zeros_like(X, dtype=np.float32)

    searchlights_iter = regions
    for sl in searchlights_iter:
        x, y = X[:, sl], Y[:, sl]
        t = sl_func(x, y)
        if return_betas:
            T[np.ix_(sl, sl)] += t
        else:
            Yhat[:, sl] += x @ t

    res = T if return_betas else Yhat
    return res


def piece_procrustes(
    X,
    Y,
    regions,
    T0=None,
    reflection=True,
    scaling=False,
):
    """
    Computes a transformation matrix from a template and a target signal using Procrustes hyperalignment.

    Parameters:
    ----------
    X : ndarray
        The source data matrix of shape (n_samples, n_features).
    Y : ndarray
        The target data matrix of shape (n_samples, n_features).
    regions : list of arrays
        List of brain regions. Contains the indices of the voxels in each region (either parcels or searchlights).
    T0 : array-like, optional
        The initial transformation matrix. Defaults to None.
    reflection : bool, optional
        Whether to allow reflection. Defaults to True.

    Returns:
    -------
    T : array-like
        The transformation matrix T.


    """
    sl_func = functools.partial(procrustes, reflection=reflection, scaling=scaling)
    T = iter_hyperalignment(
        X,
        Y,
        regions,
        T0=T0,
        sl_func=sl_func,
    )
    return T


def piece_ridge(
    X,
    Y,
    regions,
    alpha=1e3,
    verbose=False,
    return_betas=False,
):
    """
    Perform searchlight ridge regression for hyperalignment.

    Parameters:
    ----------
    X : ndarray
        The source data matrix of shape (n_samples, n_features).
    Y : ndarray
        The target data matrix of shape (n_samples, n_features).
    regions : list of arrays
        List of brain regions. Contains the indices of the voxels in each region (either parcels or searchlights).
    alpha : float(optional)
        The regularization parameter for Ridge regression. Defaults to 1e3.
    return_betas : bool(optional)
        Whether to return the coefficients of regression instead of the prediciton. Defaults to False.

    Returns:
    -------
    T : array-like
        The transformation matrix T.

    """
    sl_func = functools.partial(ridge, alpha=alpha)

    T = iter_hyperalignment(
        X,
        Y,
        regions,
        sl_func=sl_func,
        return_betas=return_betas,
    )
    return T


def template(
    X,
    regions,
    n_jobs=1,
    template_kind="pca",
    common_topography=True,
    weights=None,
):
    """
    Compute a template by aggregating local templates within searchlights.

    Parameters:
    ----------

    X : ndarray
        The input data matrix of shape (n_subjects, n_samples, n_features).
    regions : list of ndarrays
        List of regions composed of indices of voxels.
    n_jobs : int(optional)
        The number of parallel jobs to run. Defaults to 1.
    template_kind : str(optional)
        The kind of template to compute. Defaults to "pca".


    Returns:
    -------
    template : ndarray of shape (n_timepoints, n_voxels)
        The computed template.

    """
    with Parallel(n_jobs=n_jobs, batch_size=1, verbose=1) as parallel:
        local_templates = parallel(
            delayed(compute_template)(
                X,
                region=region,
                kind=template_kind,
                n_components=150,
                common_topography=common_topography,
            )
            for region in regions
        )

    template = np.zeros_like(X[0])
    if weights is not None:
        for local_template, w, region in zip(local_templates, weights, regions):
            template[:, region] += local_template * w[np.newaxis]
    else:
        for local_template, region in zip(local_templates, regions):
            template[:, region] += local_template
    return template
