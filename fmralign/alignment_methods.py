# -*- coding: utf-8 -*-
"""Module implementing alignment estimators on ndarrays."""
import warnings
import numpy as np
import scipy
from joblib import Parallel, delayed
from scipy import linalg
from scipy.optimize import linear_sum_assignment
from scipy.sparse import diags
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeCV
from sklearn.metrics.pairwise import pairwise_distances
from .hyperalignment.regions_alignment import RegionAlignment
from .hyperalignment.linalg import safe_svd, svd_pca

import jax
from ott.geometry import costs, geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


def scaled_procrustes(X, Y, scaling=False, primal=None):
    """
    Compute a mixing matrix R and a scaling sc such that Frobenius norm
    ||sc RX - Y||^2 is minimized and R is an orthogonal matrix

    Parameters
    ----------
    X: (n_samples, n_features) nd array
        source data
    Y: (n_samples, n_features) nd array
        target data
    scaling: bool
        If scaling is true, computes a floating scaling parameter sc such that:
        ||sc * RX - Y||^2 is minimized and
        - R is an orthogonal matrix
        - sc is a scalar
        If scaling is false sc is set to 1
    primal: bool or None, optional,
         Whether the SVD is done on the YX^T (primal) or Y^TX (dual)
         if None primal is used iff n_features <= n_timeframes

    Returns
    ----------
    R: (n_features, n_features) nd array
        transformation matrix
    sc: int
        scaling parameter
    """
    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)
    if np.linalg.norm(X) == 0 or np.linalg.norm(Y) == 0:
        return np.eye(X.shape[1]), 1
    if primal is None:
        primal = X.shape[0] >= X.shape[1]
    if primal:
        A = Y.T.dot(X)
        if A.shape[0] == A.shape[1]:
            A += +1.0e-18 * np.eye(A.shape[0])
        U, s, V = linalg.svd(A, full_matrices=0)
        R = U.dot(V)
    else:  # "dual" mode
        Uy, sy, Vy = linalg.svd(Y, full_matrices=0)
        Ux, sx, Vx = linalg.svd(X, full_matrices=0)
        A = np.diag(sy).dot(Uy.T).dot(Ux).dot(np.diag(sx))
        U, s, V = linalg.svd(A)
        R = Vy.T.dot(U).dot(V).dot(Vx)

    if scaling:
        sc = s.sum() / (np.linalg.norm(X) ** 2)
    else:
        sc = 1
    return R.T, sc


def optimal_permutation(X, Y):
    """
    Compute the optmal permutation matrix of X toward Y.

    Parameters
    ----------
    X: (n_samples, n_features) nd array
        source data
    Y: (n_samples, n_features) nd array
        target data

    Returns
    ----------
    permutation : (n_features, n_features) nd array
        transformation matrix
    """
    dist = pairwise_distances(X.T, Y.T)
    u = linear_sum_assignment(dist)
    u = np.array(list(zip(*u)))
    permutation = scipy.sparse.csr_matrix((np.ones(X.shape[1]), (u[:, 0], u[:, 1]))).T
    return permutation


def _projection(x, y):
    """
    Compute scalar d minimizing ||dx-y||.

    Parameters
    ----------
    x: (n_features) nd array
        source vector
    y: (n_features) nd array
        target vector

    Returns
    --------
    d: int
        scaling factor
    """
    if (x == 0).all():
        return 0
    else:
        return np.dot(x, y) / np.linalg.norm(x) ** 2


def _voxelwise_signal_projection(X, Y, n_jobs=1, parallel_backend="threading"):
    """
    Compute D, list of scalar d_i minimizing :
    ||d_i * x_i - y_i|| for every x_i, y_i in X, Y

    Parameters
    ----------
    X: (n_samples, n_features) nd array
        source data
    Y: (n_samples, n_features) nd array
        target data

    Returns
    --------
    D: list of ints
        List of optimal scaling factors
    """
    return Parallel(n_jobs, parallel_backend)(
        delayed(_projection)(voxel_source, voxel_target)
        for voxel_source, voxel_target in zip(X, Y)
    )


class Alignment(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, Y):
        pass

    def transform(self, X):
        pass


class Identity(Alignment):
    """Compute no alignment, used as baseline for benchmarks : RX = X."""

    def transform(self, X):
        """Returns X"""
        return X


class DiagonalAlignment(Alignment):
    """
    Compute the voxelwise projection factor between X and Y.

    Parameters
    ----------
    n_jobs: integer, optional (default = 1)
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.
    parallel_backend: str, ParallelBackendBase instance, None (default: 'threading')
        Specify the parallelization backend implementation. For more
        informations see joblib.Parallel documentation

    Attributes
    -----------
    R : scipy.sparse.diags
        Scaling matrix containing the optimal shrinking factor for every voxel
    """

    def __init__(self, n_jobs=1, parallel_backend="threading"):
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

    def fit(self, X, Y):
        """

        Parameters
        --------------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        """
        shrinkage_coefficients = _voxelwise_signal_projection(
            X.T, Y.T, self.n_jobs, self.parallel_backend
        )

        self.R = diags(shrinkage_coefficients)
        return

    def transform(self, X):
        """Transform X using optimal coupling computed during fit."""
        return self.R.dot(X.T).T


class ScaledOrthogonalAlignment(Alignment):
    """
    Compute a orthogonal mixing matrix R and a scaling sc.
    These are calculated such that Frobenius norm ||sc RX - Y||^2 is minimized.

    Parameters
    -----------
    scaling : boolean, optional
        Determines whether a scaling parameter is applied to improve transform.

    Attributes
    -----------
    R : ndarray (n_features, n_features)
        Optimal orthogonal transform
    """

    def __init__(self, scaling=True):
        self.scaling = scaling
        self.scale = 1

    def fit(self, X, Y):
        """
        Fit orthogonal R s.t. ||sc XR - Y||^2

        Parameters
        -----------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        """
        R, sc = scaled_procrustes(X, Y, scaling=self.scaling)
        self.scale = sc
        self.R = sc * R
        return self

    def transform(self, X):
        """Transform X using optimal transform computed during fit."""
        return X.dot(self.R)


class RidgeAlignment(Alignment):
    """
    Compute a scikit-estimator R using a mixing matrix M s.t Frobenius
    norm || XM - Y ||^2 + alpha * ||M||^2 is minimized with cross-validation

    Parameters
    ----------
    R : scikit-estimator from sklearn.linear_model.RidgeCV
        with methods fit, predict
    alpha : numpy array of shape [n_alphas]
        Array of alpha values to try. Regularization strength;
        must be a positive float. Regularization improves the conditioning
        of the problem and reduces the variance of the estimates.
        Larger values specify stronger regularization. Alpha corresponds to
        ``C^-1`` in other models such as LogisticRegression or LinearSVC.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        -None, to use the efficient Leave-One-Out cross-validation
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
    """

    def __init__(self, alphas=[0.1, 1.0, 10.0, 100, 1000], cv=4):
        self.alphas = [alpha for alpha in alphas]
        self.cv = cv

    def fit(self, X, Y):
        """
        Fit R s.t. || XR - Y ||^2 + alpha ||R||^2 is minimized with cv

        Parameters
        -----------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        """
        self.R = RidgeCV(
            alphas=self.alphas,
            fit_intercept=True,
            scoring="r2",
            cv=self.cv,
        )
        self.R.fit(X, Y)
        return self

    def transform(self, X):
        """Transform X using optimal transform computed during fit."""
        return self.R.predict(X)


class Hungarian(Alignment):
    """
    Compute the optimal permutation matrix of X toward Y

    Attributes
    ----------
    R : scipy.sparse.csr_matrix
        Mixing matrix containing the optimal permutation
    """

    def fit(self, X, Y):
        """

        Parameters
        -----------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data"""
        self.R = optimal_permutation(X, Y).T
        return self

    def transform(self, X):
        """Transform X using optimal permutation computed during fit."""
        return X.dot(self.R.toarray())


class POTAlignment(Alignment):
    """
    Compute the optimal coupling between X and Y with entropic regularization,
    using the pure Python POT (https://pythonot.github.io/) package.

    Parameters
    ----------
    solver : str (optional)
        solver from POT called to find optimal coupling 'sinkhorn',
        'greenkhorn', 'sinkhorn_stabilized','sinkhorn_epsilon_scaling', 'exact'
        see POT/ot/bregman on Github for source code of solvers
    metric : str (optional)
        metric used to create transport cost matrix,
        see full list in scipy.spatial.distance.cdist doc
    reg : int (optional)
        level of entropic regularization

    Attributes
    ----------
    R : scipy.sparse.csr_matrix
        Mixing matrix containing the optimal permutation
    """

    def __init__(
        self,
        solver="sinkhorn_epsilon_scaling",
        metric="euclidean",
        reg=1,
        max_iter=1000,
        tol=1e-3,
    ):
        self.solver = solver
        self.metric = metric
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, Y):
        """

        Parameters
        --------------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        """
        import ot

        n = len(X.T)
        if n > 5000:
            warnings.warn(
                f"One parcel is {n} voxels. As optimal transport on this region "
                "would take too much time, no alignment was performed on it. "
                "Decrease parcel size to have intended behavior of alignment."
            )
            self.R = np.eye(n)
            return self
        else:
            a = np.ones(n) * 1 / n
            b = np.ones(n) * 1 / n

            M = cdist(X.T, Y.T, metric=self.metric)

            if self.solver == "exact":
                self.R = ot.lp.emd(a, b, M) * n
            else:
                self.R = (
                    ot.sinkhorn(
                        a,
                        b,
                        M,
                        self.reg,
                        method=self.solver,
                        numItermax=self.max_iter,
                        stopThr=self.tol,
                    )
                    * n
                )
            return self

    def transform(self, X):
        """Transform X using optimal coupling computed during fit."""
        return X.dot(self.R)


class OptimalTransportAlignment(Alignment):
    """
    Compute the optimal coupling between X and Y with entropic regularization
    using a OTT-JAX as a backend for acceleration.

    Parameters
    ----------
    metric : str(optional)
        metric used to create transport cost matrix,
        see full list in scipy.spatial.distance.cdist doc
    reg : int (optional)
        level of entropic regularization

    Attributes
    ----------
    R : jaxlib.xla_extension.Array
        Mixing matrix containing the optimal permutation
    """

    def __init__(self, metric="euclidean", reg=1, max_iter=1000, tol=1e-3):
        self.metric = metric
        self.reg = reg
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, Y):
        """

        Parameters
        --------------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        """

        if self.metric == "euclidean":
            cost_matrix = costs.Euclidean().all_pairs(x=X.T, y=Y.T)
        else:
            cost_matrix = cdist(X.T, Y.T, metric=self.metric)

        geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=self.reg)
        problem = linear_problem.LinearProblem(geom)
        solver = sinkhorn.Sinkhorn(
            geom, max_iterations=self.max_iter, threshold=self.tol
        )
        P = jax.jit(solver)(problem)
        self.R = np.asarray(P.matrix * len(X.T))

        return self

    def transform(self, X):
        """Transform X using optimal coupling computed during fit."""
        return X.dot(self.R)


class IndividualizedNeuralTuning(Alignment):
    """
    Method of alignment based on the Individualized Neural Tuning model, by Feilong Ma et al. (2023).
    It works on 4D fMRI data, and is based on the assumption that the neural response to a stimulus is shared across subjects.
    It uses searchlight/parcelation alignment to denoise the data, and then computes the stimulus response matrix.
    See article : https://doi.org/10.1162/imag_a_00032
    """

    def __init__(
        self,
        template="pca",
        decomp_method=None,
        n_components=None,
        alignment_method="searchlight",
        n_jobs=1,
    ):
        """
        Initialize the IndividualizedNeuralTuning object.
        Parameters:
        --------

        - template (str): The type of template to use for alignment. Default is "pca".
        - decomp_method (str): The decomposition method to use. Default is None.
        - alignment_method (str): The alignment method to use. Can be either "searchlight" or "parcelation", Default is "searchlight".
        - n_components (int): The number of latent dimensions to use in the shared stimulus information matrix. Default is None.
        - n_jobs (int): The number of parallel jobs to run. Default is -1.

        Returns:
        --------
        None
        """

        self.n_s = None
        self.n_t = None
        self.n_v = None
        self.labels = None
        self.alphas = None
        self.alignment_method = alignment_method
        if alignment_method == "parcelation":
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
        self.tmpl_kind = template
        self.latent_dim = n_components
        self.n_jobs = n_jobs

    #######################################################################################
    # Computing decomposition

    @staticmethod
    def _tuning_estimator(shared_response, target):
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
        return np.linalg.pinv(shared_response).dot(target).astype(np.float32)

    @staticmethod
    def _stimulus_estimator(full_signal, n_t, n_s, latent_dim=None):
        """
        Estimates the stimulus response using the given parameters.

        Args:
            full_signal (np.ndarray): The full signal data.
            n_t (int): The number of time points.
            n_s (int): The number of stimuli.
            latent_dim (int, optional): The number of latent dimensions. Defaults to None.

        Returns:
            stimulus (np.ndarray): The stimulus response of shape (n_t, latent_dim) or (n_t, n_t).
        """
        if latent_dim is not None and latent_dim < n_t:
            U = svd_pca(full_signal)
            U = U[:, :latent_dim]
        else:
            U, _, _ = safe_svd(full_signal)

        stimulus = np.sqrt(n_s) * U
        stimulus = stimulus.astype(np.float32)
        return stimulus

    @staticmethod
    def _reconstruct_signal(shared_response, individual_tuning):
        """
        Reconstructs the signal using the shared response and individual tuning.

        Args:
            shared_response (numpy.ndarray): The shared response of shape (n_t, n_t) or (n_t, latent_dim).
            individual_tuning (numpy.ndarray): The individual tuning of shape (latent_dim, n_v) or (n_t, n_v).

        Returns:
            numpy.ndarray: The reconstructed signal of shape (n_t, n_v) (same shape as the original signal)
        """
        return (shared_response @ individual_tuning).astype(np.float32)

    def fit(
        self,
        X_train,
        searchlights=None,
        parcels=None,
        dists=None,
        radius=20,
        tuning=True,
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
        - tuning (bool, optional):
            Whether to compute the tuning weights. Defaults to True.
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

        self.n_s, self.n_t, self.n_v = X_train_.shape

        self.tuning_data = np.empty(self.n_s, dtype=np.float32)
        self.denoised_signal = np.empty(self.n_s, dtype=np.float32)

        if searchlights is None:
            self.regions = parcels
        else:
            self.regions = searchlights
            self.distances = dists
            self.radius = radius

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

        # Stimulus matrix computation
        if self.decomp_method is None:
            full_signal = np.concatenate(self.denoised_signal, axis=1)
            self.shared_response = self._stimulus_estimator(
                full_signal, self.n_t, self.n_s, self.latent_dim
            )
            if tuning:
                self.tuning_data = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._tuning_estimator)(
                        self.shared_response,
                        self.denoised_signal[i],
                    )
                    for i in range(self.n_s)
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
            stimulus_ = self._stimulus_estimator(
                full_signal, self.n_t, self.n_s, self.latent_dim
            )

        if verbose:
            print("Predict : stimulus matrix shape: ", stimulus_.shape)

        reconstructed_signal = [
            self._reconstruct_signal(stimulus_, T_est) for T_est in self.tuning_data
        ]
        return np.array(reconstructed_signal, dtype=np.float32)
