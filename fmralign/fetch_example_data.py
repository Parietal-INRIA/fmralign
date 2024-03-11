# -*- coding: utf-8 -*-
import os

import pandas as pd
from nilearn.datasets._utils import fetch_files, get_dataset_dir
from fastsrm.srm import projection
import numpy as np


def fetch_ibc_subjects_contrasts(subjects, data_dir=None, verbose=1):
    """Fetch all IBC contrast maps for each of subjects.
    After downloading all relevant images that are not already cached,
    it returns a dataframe with all needed links.

    Parameters
    ----------
    subjects : list of str.
        Subjects data to download. Available strings are ['sub-01', 'sub-02',
        'sub-04' ... 'sub-09', 'sub-11' ... sub-15]
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location.
    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
    files : list of list of str
        List (for every subject) of list of path (for every conditions),
        in ap then pa acquisition.
    metadata_df : Pandas Dataframe
        Table containing some metadata for each available image in the dataset,
        as well as their pathself.
        Filtered to contain only the 'subjects' parameter metadatas
    mask: str
        Path to the mask to be used on the data
    Notes
    ------
    This function is a caller to nilearn.datasets._utils.fetch_files in order
    to simplify examples reading and understanding for fmralign.
    See Also
    ---------
    nilearn.datasets.fetch_localizer_calculation_task
    nilearn.datasets.fetch_localizer_contrasts
    """
    # The URLs can be retrieved from the nilearn account on OSF
    if subjects == "all":
        subjects = ["sub-{i:02d}" for i in [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]
    dataset_name = "ibc"
    data_dir = get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

    # download or retrieve metadatas, put it in a dataframe,
    # list all condition and specify path to the right directory
    metadata_path = fetch_files(
        data_dir,
        [
            (
                "ibc_3mm_all_subjects_metadata.csv",
                "https://osf.io/pcvje/download",
                {"uncompress": True},
            )
        ],
        verbose=verbose,
    )
    metadata_df = pd.read_csv(metadata_path[0])
    conditions = metadata_df.condition.unique()
    metadata_df["path"] = metadata_df["path"].str.replace("path_to_dir", data_dir)
    # filter the dataframe to return only rows relevant for subjects argument
    metadata_df = metadata_df[metadata_df.subject.isin(subjects)]

    # download / retrieve mask niimg and find its path
    mask = fetch_files(
        data_dir,
        [
            (
                "gm_mask_3mm.nii.gz",
                "https://osf.io/yvju3/download",
                {"uncompress": True},
            )
        ],
        verbose=verbose,
    )[0]

    # list all url keys for downloading separetely each subject data
    url_keys = {
        "sub-01": "8z23h",
        "sub-02": "e9kbm",
        "sub-04": "qn5b6",
        "sub-05": "u74a3",
        "sub-06": "83bje",
        "sub-07": "43j69",
        "sub-08": "ua8qx",
        "sub-09": "bxwtv",
        "sub-11": "3dfbv",
        "sub-12": "uat7d",
        "sub-13": "p238h",
        "sub-14": "prdk4",
        "sub-15": "sw72z",
    }

    # for all subjects in argument, download all contrasts images and list
    # their path in the variable files
    opts = {"uncompress": True}
    files = []
    for subject in subjects:
        url = f"https://osf.io/{url_keys[subject]}/download"
        filenames = [
            (os.path.join(subject, f"{condition}_ap.nii.gz"), url, opts)
            for condition in conditions
        ]
        filenames.extend(
            [
                (os.path.join(subject, f"{condition}_pa.nii.gz"), url, opts)
                for condition in conditions
            ]
        )
        files.append(fetch_files(data_dir, filenames, verbose=verbose))
    return files, metadata_df, mask


def generate_dummy_signal(
    n_subjects: int,
    n_timepoints: int,
    n_voxels: int,
    S_std=1,
    latent_dim=None,
    T_mean=0,
    T_std=1,
    SNR=1,
    generative_method="custom",
    seed=0,
):
    """Generate dummy signal for testing INT model

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_timepoints : int
        Number of timepoints.
    n_voxels : int
        Number of voxels.
    S_std : float, default=1
        Standard deviation of latent variables.
    latent_dim: int, defult=None
        Number of latent dimensions. Defualts to n_timepoints
    T_mean : float
        Mean of weights.
    T_std : float
        Standard deviation of weights.
    SNR : float
        Signal-to-noise ratio.
    generative_method : str, default="custom"
        Method for generating data. Options are "custom", "fastsrm".
    seed : int
        Random seed.


    Returns
    -------
    imgs_train : ndarray of shape (n_subjects, n_timepoints, n_voxels)
        Training data.
    imgs_test : ndarray of shape (n_subjects, n_timepoints, n_voxels)
        Testing data.
    S_train : ndarray of shape (n_timepoints, latent_dim)
        Training latent variables.
    S_test : ndarray of shape (n_timepoints, latent_dim)
        Testing latent variables.
    Ts : ndarray of shape (n_subjects, latent_dim , n_voxels)
        Tuning matrices.
    """
    if latent_dim is None:
        latent_dim = n_timepoints

    rng = np.random.RandomState(seed=seed)

    if generative_method == "custom":
        sigma = n_subjects * np.arange(1, latent_dim + 1)
        # np.random.shuffle(sigma)
        # Generate common signal matrix
        S_train = S_std * np.random.randn(n_timepoints, latent_dim)
        # Normalize each row to have unit norm
        S_train = S_train / np.linalg.norm(S_train, axis=0, keepdims=True)
        S_train = S_train @ np.diag(sigma)
        S_test = S_std * np.random.randn(n_timepoints, latent_dim)
        S_test = S_test / np.linalg.norm(S_test, axis=0, keepdims=True)
        S_test = S_test @ np.diag(sigma)

    elif generative_method == "fastsrm":
        Sigma = rng.dirichlet(np.ones(latent_dim), 1).flatten()
        S_train = np.sqrt(Sigma)[:, None] * rng.randn(n_timepoints, latent_dim)
        S_test = np.sqrt(Sigma)[:, None] * rng.randn(n_timepoints, latent_dim)

    elif generative_method == "multiviewica":
        S_train = np.random.laplace(size=(n_timepoints, latent_dim))
        S_test = np.random.laplace(size=(n_timepoints, latent_dim))

    else:
        raise ValueError("Unknown generative method")

    # Generate indiivdual spatial components
    data_train, data_test = [], []
    Ts = []
    for _ in range(n_subjects):
        if generative_method == "custom" or generative_method == "multiviewica":
            W = T_mean + T_std * np.random.randn(latent_dim, n_voxels)
        else:
            W = projection(rng.randn(latent_dim, n_voxels))

        Ts.append(W)
        X_train = S_train @ W
        noise = np.random.randn(n_timepoints, n_voxels)
        noise = (
            noise
            * np.linalg.norm(X_train)
            / (SNR * np.linalg.norm(noise, axis=0, keepdims=True))
        )
        X_train += noise
        data_train.append(X_train)
        X_test = S_test @ W
        noise = np.random.randn(n_timepoints, n_voxels)
        noise = (
            noise
            * np.linalg.norm(X_test)
            / (SNR * np.linalg.norm(noise, axis=0, keepdims=True))
        )
        X_test += noise
        data_test.append(X_test)

    data_train = np.array(data_train)
    data_test = np.array(data_test)
    return data_train, data_test, S_train, S_test, Ts


def generate_dummy_searchlights(
    n_searchlights: int,
    n_voxels: int,
    radius: float,
    sl_size: int = 5,
    seed: int = 0,
):
    """Generate dummy searchlights for testing INT model

    Parameters
    ----------
    n_searchlights : int
        Number of searchlights.
    n_voxels : int
        Number of voxels.
    radius : float,
        Radius of searchlights.
    sl_size : int, default=5
        Size of each searchlight (easier for dummy signal generation).
    seed : int
        Random seed.

    Returns
    -------
    searchlights : ndarray of shape (n_searchlights, sl_size)
        Searchlights.
    dists : ndarray of shape (n_searchlights, sl_size)
        Distances.
    """
    rng = np.random.RandomState(seed=seed)
    searchlights = rng.randint(n_voxels, size=(n_searchlights, sl_size))
    dists = rng.randint(radius, size=searchlights.shape)
    return searchlights, dists
