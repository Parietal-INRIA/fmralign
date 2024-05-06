# -*- coding: utf-8 -*-
"""Module implementing alignment estimators on ndarrays."""
import warnings

import numpy as np
import torch
import scipy
from joblib import Parallel, delayed
from scipy import linalg
import ot
from scipy.optimize import linear_sum_assignment
from scipy.sparse import diags
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeCV
from sklearn.metrics.pairwise import pairwise_distances
from nilearn import masking

# Fast implementation for parallelized computing
from fmralign.hyperalignment.linalg import safe_svd, svd_pca
from fmralign.hyperalignment.piecewise_alignment import PiecewiseAlignment

from fugw.mappings import FUGW, FUGWSparse
from fugw.scripts import coarse_to_fine, lmds


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
    permutation = scipy.sparse.csr_matrix(
        (np.ones(X.shape[1]), (u[:, 0], u[:, 1]))
    ).T
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
        ----------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        """

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
        ----------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        """
        import jax
        from ott.geometry import costs, geometry
        from ott.problems.linear import linear_problem
        from ott.solvers.linear import sinkhorn

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
    Method of alignment based on the Individualized Neural Tuning model.
    It works on 4D fMRI data, and is based on the assumption that the neural
    response to a stimulus is shared across subjects. It uses searchlight/
    parcelation alignment to denoise the data, and then computes the stimulus
    response matrix.
    See article : https://doi.org/10.1162/imag_a_00032
    """

    def __init__(
        self,
        decomp_method="pca",
        n_components=None,
        searchlights=None,
        parcels=None,
        dists=None,
        radius=20,
        tuning=True,
        n_jobs=1,
    ):
        """
        Initialize the IndividualizedNeuralTuning object.

        Parameters:
        ----------
        decomp_method : str
             The decomposition method to use.
             Can be ["pca", "pcav1", "procrustes"]
             Default is "pca".
        searchlights : array-like
            The searchlight indices for each subject,
            of shape (n_s, n_searchlights).
        parcels : array-like
            The parcel indices for each subject,
            of shape (n_s, n_parcels) (if not using searchlights)
        dists : array-like
            The distances of vertices to the center of their searchlight,
            of shape (n_searchlights, n_vertices_sl)
        radius : int(optional)
            The radius of the searchlight sphere, in milimeters.
            Defaults to 20.
        tuning : bool(optional)
            Whether to compute the tuning weights. Defaults to True.
        n_components : int
             The number of latent dimensions to use in the shared stimulus
             information
             matrix. Default is None.
        n_jobs : int
             The number of parallel jobs to run. Default is -1.

        Returns:
        --------
        None
        """

        self.n_subjects = None
        self.n_time_points = None
        self.labels = None
        self.alphas = None

        if searchlights is None and parcels is None:
            raise ValueError("searchlights or parcels must be provided")

        if searchlights is not None and parcels is not None:
            raise ValueError(
                "searchlights and parcels cannot be provided at the same time"
            )

        if searchlights is not None:
            self.regions = searchlights
        else:
            self.regions = parcels

        self.dists = dists
        self.radius = radius
        self.tuning = tuning

        self.tuning_data = []
        self.denoised_signal = []
        self.decomp_method = decomp_method
        self.n_components = n_components
        self.n_jobs = n_jobs

    ################################################################
    # Computing decomposition

    @staticmethod
    def _tuning_estimator(shared_response, target):
        """
        Estimate the tuning matrix for individualized neural tuning.

        Parameters:
        ----------
        shared_response : array-like
             The shared response matrix of shape (n_timepoints, k)
             where k is the dimension of the sources latent space.
        target : array-like
             The target matrix.
        latent_dim : int, optional
             The number of latent dimensions (if PCA is used). Defaults to None.

        Returns:
        --------
        array-like: The estimated tuning matrix for the given target.

        """
        if shared_response.shape[1] != shared_response.shape[0]:
            return (np.linalg.pinv(shared_response)).dot(target)
        return np.linalg.inv(shared_response).dot(target)

    @staticmethod
    def _stimulus_estimator(
        full_signal, n_subjects, latent_dim=None, scaling=True
    ):
        """
        Estimates the stimulus matrix for the Individualized Neural Tuning model.

        Parameters:
        -----------
        full_signal : ndarray
             Concatenated signal for all subjects,
             of shape (n_timepoints, n_subjects * n_voxels).
        n_subjects : int
             The number of subjects.
        latent_dim : int, optional
             The number of latent dimensions to use. Defaults to None.
        scaling : bool, optional
             Whether to scale the stimulus matrix sources. Defaults to True.

        Returns:
        --------
        stimulus : ndarray
            The stimulus matrix of shape (n_timepoints, n_subjects * n_voxels)
        """
        n_timepoints = full_signal.shape[0]
        if scaling:
            U = svd_pca(full_signal)
        else:
            U, _, _ = safe_svd(full_signal)
        if latent_dim is not None and latent_dim < n_timepoints:
            U = U[:, :latent_dim]

        stimulus = np.sqrt(n_subjects) * U
        stimulus = stimulus.astype(np.float32)
        return stimulus

    @staticmethod
    def _reconstruct_signal(shared_response, individual_tuning):
        """
        Reconstructs the signal using the stimulus as shared
        response and individual tuning.

        Parameters:
        --------
        shared_response : ndarray
             The shared response of shape (n_timeframes, n_timeframes) or
             (n_timeframes, latent_dim).
        individual_tuning : ndarray
             The individual tuning of shape (latent_dim, n_voxels) or
             (n_timeframes, n_voxels).

        Returns:
        --------
        ndarray:
            The reconstructed signal of shape (n_timeframes, n_voxels).
        """
        return (shared_response @ individual_tuning).astype(np.float32)

    def fit(
        self,
        X,
        verbose=True,
    ):
        """
        Fits the IndividualizedNeuralTuning model to the training data.

        Parameters:
        -----------
        X : array-like
            The training data of shape (n_subjects, n_samples, n_voxels).
        verbose : bool(optional)
            Whether to print progress information. Defaults to True.

        Returns:
        --------

        self : Instance of IndividualizedNeuralTuning)
            The fitted model.
        """

        X_ = np.array(X, copy=True, dtype=np.float32)

        self.n_subjects, self.n_time_points, self.n_voxels = X_.shape

        self.tuning_data = np.empty(self.n_subjects, dtype=np.float32)
        self.denoised_signal = np.empty(self.n_subjects, dtype=np.float32)

        denoiser = PiecewiseAlignment(
            template_kind=self.decomp_method,
            n_jobs=self.n_jobs,
            verbose=verbose,
        )
        self.denoised_signal = denoiser.fit_transform(
            X_,
            regions=self.regions,
            dists=self.dists,
            radius=self.radius,
        )

        # Stimulus matrix computation
        full_signal = np.concatenate(self.denoised_signal, axis=1)
        self.shared_response = self._stimulus_estimator(
            full_signal, self.n_subjects, self.n_components
        )
        if self.tuning:
            self.tuning_data = Parallel(n_jobs=self.n_jobs)(
                delayed(self._tuning_estimator)(
                    self.shared_response,
                    self.denoised_signal[i],
                )
                for i in range(self.n_subjects)
            )

        return self

    def transform(self, X, verbose=False):
        """
        Transforms the input test data using the hyperalignment model.

        Parameters:
        ----------
        X : array-like
            The test data of shape (n_subjects, n_timepoints, n_voxels).
        verbose : bool(optional)
            Whether to print progress information. Defaults to False.

        Returns:
        --------
        ndarray :
            The transformed data of shape (n_subjects, n_timepoints, n_voxels).
        """

        full_signal = np.concatenate(X, axis=1, dtype=np.float32)

        if verbose:
            print("Predict : Computing stimulus matrix...")

        stimulus_ = self._stimulus_estimator(
            full_signal, self.n_subjects, self.n_components
        )
        print("Predict : stimulus matrix shape: ", stimulus_.shape)

        if verbose:
            print("Predict : stimulus matrix shape: ", stimulus_.shape)

        reconstructed_signal = Parallel(n_jobs=self.n_jobs)(
            delayed(self._reconstruct_signal)(stimulus_, T_est)
            for T_est in self.tuning_data
        )

        return np.array(reconstructed_signal, dtype=np.float32)


class FugwAlignment:
    """Wrapper for FUGW alignment"""

    def __init__(
        self,
        alpha_coarse=0.5,
        alpha_fine=0.5,
        rho_coarse=1.0,
        rho_fine=1.0,
        eps_coarse=1.0,
        eps_fine=1.0,
        radius=5,
        anisotropy=(3, 3, 3),
        reg_mode="independent",
        divergence="kl",
    ) -> None:
        """Initialize FUGW alignment

        Parameters
        ----------
        n_samples : int, optional
            Number of samples from the embedding, by default 300.
        alpha_coarse : float, optional, by default 0.5.
        rho_coarse : float, optional, by default 1.
        eps_coarse : float, optional, by default 1.
        alpha_fine : float, optional, by default 0.5.
        rho_fine : float, optional, by default 1.
        eps_fine : float, optional, by default 1e-6.
        radius : int, optional
            Radius around the sampled points in mm, by default 5.
        anisotropy : tuple, optional.
            Anisotropy of the fmri mask, by default (3, 3, 3)
        reg_mode : str, optional
            Regularization mode, by default "independent"
        divergence : str, optional
            Divergence used in the FUGW alignment, by default "kl".
        """
        self.alpha_coarse = alpha_coarse
        self.rho_coarse = rho_coarse
        self.eps_coarse = eps_coarse
        self.alpha_fine = alpha_fine
        self.rho_fine = rho_fine
        self.eps_fine = eps_fine
        self.radius = radius
        self.anisotropy = anisotropy
        self.reg_mode = reg_mode
        self.divergence = divergence

    def _normalize(self, X):
        """Normalize the input data"""
        return (X / np.linalg.norm(X, axis=1).reshape(-1, 1)).T

    def _prepare_geometry_embedding(self, segmentation, n_landmarks, verbose):
        """Compute the normalized geometry embedding"""
        geometry_embedding = lmds.compute_lmds_volume(
            segmentation,
            k=12,
            n_landmarks=n_landmarks,
            anisotropy=self.anisotropy,
            verbose=verbose,
        ).nan_to_num()

        (
            geometry_embedding_normalized,
            max_distance,
        ) = coarse_to_fine.random_normalizing(geometry_embedding)

        return (
            geometry_embedding,
            geometry_embedding_normalized,
            max_distance,
        )

    def _sample_geometry(self, segmentation, geometry_embedding, n_samples):
        """Sample the geometry of the mask"""
        return coarse_to_fine.sample_volume_uniformly(
            segmentation,
            embeddings=geometry_embedding,
            n_samples=n_samples,
        )

    def fit(
        self,
        X,
        Y,
        segmentation,
        method="coarse-to-fine",
        n_samples=10,
        n_landmarks=100,
        device="auto",
        verbose=True,
        **kwargs,
    ):
        """Fit FUGW alignment

        Parameters
        ----------
        X : ndarray
            Source features
        Y : ndarray
            Target features
        segmentation : ndarray,
            Segmentation of the mask
        method : str, optional
            Method used to compute FUGW alignments, by default "coarse_to_fine".
        n_samples : int, optional
            Number of samples from the embedding, by default 300.
        n_landmarks : int, optional
            Number of landmarks used in the embedding, by default 1000.
        device : torch.device, optional, by default "auto"
            Device on which to perform the computation.
        verbose : bool, optional, by default True
        **kwargs : dict
            Additional parameters passed to the FUGW mapping.fit method.

        Returns
        -------
        self : FugwAlignment
            Fitted FUGW alignment
        """
        if device == "auto":
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )

        # Compute the embedding of the source and target data
        geometry_embedding, geometry_embedding_normalized, max_distance = (
            self._prepare_geometry_embedding(
                segmentation, n_landmarks, verbose
            )
        )

        if verbose:
            print("Geometry embedding computed")

        source_features_normalized = self._normalize(X)
        target_features_normalized = self._normalize(Y)

        if verbose:
            print("Features computed")

        if method == "dense":
            mapping = FUGW(
                alpha=self.alpha_coarse,
                rho=self.rho_coarse,
                eps=self.eps_coarse,
                reg_mode=self.reg_mode,
                divergence=self.divergence,
            )

            mapping.fit(
                source_features=source_features_normalized,
                target_features=target_features_normalized,
                source_geometry=geometry_embedding_normalized
                @ geometry_embedding_normalized.T,
                target_geometry=geometry_embedding_normalized
                @ geometry_embedding_normalized.T,
                verbose=verbose,
                **kwargs,
            )

            self.mapping = mapping

        elif method == "coarse-to-fine":
            # Subsample vertices as uniformly as possible on the surface
            sampled_geometry = self._sample_geometry(
                segmentation, geometry_embedding, n_samples
            )

            if verbose:
                print("Samples computed")

            coarse_mapping = FUGW(
                alpha=self.alpha_coarse,
                rho=self.rho_coarse,
                eps=self.eps_coarse,
                reg_mode=self.reg_mode,
                divergence=self.divergence,
            )

            fine_mapping = FUGWSparse(
                alpha=self.alpha_fine,
                rho=self.rho_fine,
                eps=self.eps_fine,
                reg_mode=self.reg_mode,
                divergence=self.divergence,
            )

            coarse_to_fine.fit(
                source_features=source_features_normalized,
                target_features=target_features_normalized,
                source_geometry_embeddings=geometry_embedding_normalized,
                target_geometry_embeddings=geometry_embedding_normalized,
                source_sample=sampled_geometry,
                target_sample=sampled_geometry,
                coarse_mapping=coarse_mapping,
                source_selection_radius=(self.radius / max_distance),
                target_selection_radius=(self.radius / max_distance),
                fine_mapping=fine_mapping,
                device=device,
                verbose=verbose,
                **kwargs,
            )

            self.mapping = fine_mapping

        return self

    def transform(
        self,
        X,
        device="auto",
        id_reg=0.0,
    ):
        """Project features using the fitted FUGW alignment

        Parameters
        ----------
        X : ndarray
            Source features
        device : torch.device, optional, by default "auto"
            Device on which to perform the computation.
        id_reg: float, in the [0, 1] interval, defaults to 0
            If source/target share the same geometry,
            interpolate the transport plan with the identity
            using the provided coefficient.
            A value of 1 (resp. 0) will rely solely on the identity
            (resp. the transport plan).

        Returns
        -------
        ndarray
            Projected features
        """

        if self.mapping is None:
            raise ValueError(
                "FUGW alignment must be fitted before transforming data"
            )

        if device == "auto":
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )

        # If id_reg is True, interpolate the resulting
        # mapping with the identity matrix
        transformed_features = self.mapping.transform(
            X.T, id_reg=id_reg, device=device
        )
        return transformed_features.T
