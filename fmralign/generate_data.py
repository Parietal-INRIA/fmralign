"""Some functions to generate toy fMRI data using shared stimulus information."""

import numpy as np
from fastsrm.srm import projection


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
    radius: int,
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
    radius : int
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
