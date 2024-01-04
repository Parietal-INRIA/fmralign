import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from .regions_alignment import RegionAlignment
from .linalg import safe_svd, svd_pca
from joblib import Parallel, delayed
import os


class INT(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        tmpl_kind="pca",
        decomp_method=None,
        latent_dim=None,
        alignment_method="searchlight",
        id=None,
        cache=True,
        n_jobs=-1,
    ):
        """
        Initialize the IndividualizedNeuralTuning object.

        Parameters:
        --------

        - tmpl_kind (str): The type of template to use for alignment. Default is "pca".
        - decomp_method (str): The decomposition method to use. Default is None.
        - latent_dim (int): The number of latent dimensions to use in the shared stimulus information matrix. Default is None.
        - n_jobs (int): The number of parallel jobs to run. Default is -1.

        Returns:
        --------
        None
        """

        self.n_subjects = None
        self.n_timepoints = None
        self.n_voxels = None
        self.labels = None
        self.alphas = None
        self.alignment_method = alignment_method
        if alignment_method == "parcel":
            self.parcels = None

        elif (
            alignment_method == "searchlight"
            or alignment_method == "ensemble_searchlight"
        ):
            self.searchlights = None
            self.distances = None
            self.radius = None

        self.path = None
        self.tuning_data = []
        self.denoised_signal = []
        self.decomp_method = decomp_method
        self.tmpl_kind = tmpl_kind
        self.latent_dim = latent_dim
        self.n_jobs = n_jobs
        self.cache = cache

        if cache:
            if id is None:
                self.id = "default"
            else:
                self.id = id

            path = os.path.join(os.getcwd(), f"cache/int/{self.id}")
            # Check if cache folder exists
            if not os.path.exists(path):
                os.makedirs(path)

            self.path = path

    def fit(
        self,
        X_train,
        searchlights=None,
        parcels=None,
        dists=None,
        radius: int = 20,
        verbose=True,
    ):
        """
        Fits the IndividualizedNeuralTuning model to the training data.

        Parameters:
        --------

        - X_train (array-like):
            The training data of shape (n_subjects, n_samples, n_voxels).
        - searchlights (array-like):
            The searchlight indices for each subject, of shape (n_s, n_searchlights).
        - parcels (array-like):
            The parcel indices for each subject, of shape (n_s, n_parcels) (if not using searchlights)
        - dists (array-like):
            The distances of vertices to the center of their searchlight, of shape (n_searchlights, n_vertices_sl)
        - radius (int, optional):
            The radius of the searchlight sphere, in milimeters. Defaults to 20.
        - verbose (bool, optional):
            Whether to print progress information. Defaults to True.
        - id (str, optional):
            An identifier for caching purposes. Defaults to None.

        Returns:
        --------

        - self (IndividualizedNeuralTuning):
            The fitted model.
        """

        X_train_ = np.array(X_train, copy=True, dtype=np.float32)

        self.n_subjects, self.n_timepoints, self.n_voxels = X_train_.shape

        self.tuning_data = np.empty(self.n_subjects, dtype=np.float32)
        self.denoised_signal = np.empty(self.n_subjects, dtype=np.float32)

        if searchlights is None:
            self.regions = parcels
        else:
            self.regions = searchlights
            self.distances = dists
            self.radius = radius

        # check for cached data
        try:
            self.denoised_signal = np.load(self.path + "/train_data_denoised.npy")
            if verbose:
                print("Loaded denoised data from cache")

        except:  # noqa: E722
            denoiser = RegionAlignment(
                alignment_method=self.alignment_method,
                n_jobs=self.n_jobs,
                verbose=verbose,
                path=self.path,
            )
            self.denoised_signal = denoiser.fit_transform(
                X_train_,
                regions=self.regions,
                dists=dists,
                radius=radius,
            )
            # Clear memory of the SearchlightAlignment object
            denoiser = None

        iterm = range(self.n_subjects)

        if verbose:
            iterm = tqdm(iterm)

        # Stimulus matrix computation
        if self.decomp_method is None:
            full_signal = np.concatenate(self.denoised_signal, axis=1)
            self.shared_response = stimulus_estimator(
                full_signal, self.n_timepoints, self.n_subjects, self.latent_dim
            )

            self.tuning_data = Parallel(n_jobs=self.n_jobs)(
                delayed(tuning_estimator)(
                    self.shared_response,
                    self.denoised_signal[i],
                    latent_dim=self.latent_dim,
                )
                for i in iterm
            )

        return self

    def transform(self, X_test_, verbose=False):
        """
        Transforms the input test data using the hyperalignment model.

        Args:
            X_test_ (list of arrays):
                The input test data.
            verbose (bool, optional):
                Whether to print verbose output. Defaults to False.
            id (int, optional):
                Identifier for the transformation. Defaults to None.

        Returns:
            numpy.ndarray: The transformed test data.
        """

        full_signal = np.concatenate(X_test_, axis=1, dtype=np.float32)

        if verbose:
            print("Predict : Computing stimulus matrix...")

        if self.decomp_method is None:
            stimulus_ = stimulus_estimator(
                full_signal, self.n_timepoints, self.n_subjects, self.latent_dim
            )

        if verbose:
            print("Predict : stimulus matrix shape: ", stimulus_.shape)

        reconstructed_signal = [
            reconstruct_signal(stimulus_, T_est) for T_est in self.tuning_data
        ]
        return np.array(reconstructed_signal, dtype=np.float32)

    def clean_cache(self, id):
        """
        Removes the cache file associated with the given ID.

        Args:
            id (int): The ID of the cache file to be removed.
        """
        try:
            os.remove("cache")
        except:  # noqa: E722
            print("No cache to remove")


#######################################################################################
# Computing decomposition


def tuning_estimator(shared_response, target, latent_dim=None):
    """
    Estimate the tuning weights for individualized neural tuning.

    Parameters:
    --------
    - shared_response (array-like):
        The shared response matrix.
    - target (array-like):
        The target matrix.
    - latent_dim (int, optional):
        The number of latent dimensions. Defaults to None.

    Returns:
    --------
        array-like: The estimated tuning weights.

    """
    if latent_dim is None:
        return np.linalg.inv(shared_response).dot(target)
    return np.linalg.pinv(shared_response).dot(target).astype(np.float32)


def stimulus_estimator(full_signal, n_t, n_s, latent_dim=None):
    """
    Estimates the stimulus response using the given parameters.

    Args:
        full_signal (np.ndarray): The full signal data.
        n_t (int): The number of time points.
        n_s (int): The number of stimuli.
        latent_dim (int, optional): The number of latent dimensions. Defaults to None.

    Returns:
        np.ndarray: The estimated shared response.
    """
    if latent_dim is not None and latent_dim < n_t:
        U = svd_pca(full_signal)
        U = U[:, :latent_dim]
    else:
        U, _, _ = safe_svd(full_signal)

    shared_response = np.sqrt(n_s) * U
    return shared_response.astype(np.float32)


def reconstruct_signal(shared_response, individual_tuning):
    """
    Reconstructs the signal using the shared response and individual tuning.

    Args:
        shared_response (numpy.ndarray): The shared response of shape (n_t, n_t) or (n_t, latent_dim).
        individual_tuning (numpy.ndarray): The individual tuning of shape (latent_dim, n_v) or (n_t, n_v).

    Returns:
        numpy.ndarray: The reconstructed signal of shape (n_t, n_v) (same shape as the original signal)
    """
    return (shared_response @ individual_tuning).astype(np.float32)
