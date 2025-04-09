# -*- coding: utf-8 -*-
"""Module implementing alignment estimators on ndarrays."""

import warnings

import numpy as np
import ot
import torch
from fugw.solvers.utils import (
    batch_elementwise_prod_and_sum,
    crow_indices_to_row_indices,
    solver_sinkhorn_stabilized_sparse,
    solver_sinkhorn_eps_scaling_sparse,
)
from fugw.utils import _low_rank_squared_l2, _make_csr_matrix
from joblib import Parallel, delayed
from scipy import linalg
from scipy.sparse import diags
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeCV

# Fast implementation for parallelized computing
from fmralign.hyperalignment.linalg import safe_svd, svd_pca
from fmralign.hyperalignment.piecewise_alignment import PiecewiseAlignment


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
        reg=1e-1,
        max_iter=100,
        tol=0,
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
    Compute the (unbalanced) optimal coupling between X and Y
    with entropic regularization using
    OTT-JAX as a backend for acceleration.

    Parameters
    ----------
    metric : str(optional)
        metric used to create transport cost matrix,
        see full list in scipy.spatial.distance.cdist doc
    reg : int (optional)
        level of entropic regularization
    tau : float (optional)
        level of unbalancing, 1.0 corresponds to balanced transport,
        lower values will favor lower mass transport

    Attributes
    ----------
    R : jaxlib.xla_extension.Array
        Mixing matrix containing the optimal permutation
    """

    def __init__(
        self, metric="euclidean", reg=1e-1, tau=1.0, max_iter=100, tol=0
    ):
        self.metric = metric
        self.reg = reg
        self.tau = tau
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
        problem = linear_problem.LinearProblem(
            geom, tau_a=self.tau, tau_b=self.tau
        )
        solver = sinkhorn.Sinkhorn(
            geom, max_iterations=self.max_iter, threshold=self.tol
        )
        P = jax.jit(solver)(problem)
        self.R = np.asarray(P.matrix) * len(X.T)

        return self

    def transform(self, X):
        """Transform X using optimal coupling computed during fit."""
        return X @ self.R


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


class SparseOT(Alignment):
    """
    Compute the unbalanced regularized optimal coupling between X and Y,
    with sparsity constraints inspired by the FUGW package sparse
    sinkhorn solver.
    (https://github.com/alexisthual/fugw/blob/main/src/fugw/solvers/sparse.py)

    Parameters
    ----------
    sparsity_mask : sparse torch.Tensor of shape (n_features, n_features)
        Sparse mask that defines the sparsity pattern of the coupling matrix.
    rho : float (optional)
        Strength of the unbalancing constraint. Lower values will favor lower
        mass transport. Defaults to infinity.
    reg : float (optional)
        Strength of the entropic regularization. Defaults to 0.1.
    max_iter : int (optional)
        Maximum number of iterations. Defaults to 1000.
    tol : float (optional)
        Tolerance for stopping criterion. Defaults to 1e-7.
    eval_freq : int (optional)
        Frequency of evaluation of the stopping criterion. Defaults to 10.
    device : str (optional)
        Device on which to perform computations. Defaults to 'cpu'.
    verbose : bool (optional)
        Whether to print progress information. Defaults to False.

    Attributes
    ----------
    pi : sparse torch.Tensor of shape (n_features, n_features)
        Sparse coupling matrix
    """

    def __init__(
        self,
        sparsity_mask,
        solver="sinkhorn_epsilon_scaling",
        reg=1e-1,
        max_iter=100,
        tol=0,
        eval_freq=10,
        device="cpu",
        verbose=False,
    ):
        self.reg = reg
        self.sparsity_mask = sparsity_mask
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.eval_freq = eval_freq
        self.device = device
        self.verbose = verbose

    def _initialize_plan(self, n):
        return (
            torch.sparse_coo_tensor(
                self.sparsity_mask.indices(),
                torch.ones_like(self.sparsity_mask.values())
                / self.sparsity_mask.values().shape[0],
                (n, n),
            )
            .coalesce()
            .to_sparse_csr()
            .to(self.device)
        )

    def _cost(self, init_plan, F, n):
        crow_indices, col_indices = (
            init_plan.crow_indices(),
            init_plan.col_indices(),
        )
        row_indices = crow_indices_to_row_indices(crow_indices)
        cost_values = batch_elementwise_prod_and_sum(
            F[0], F[1], row_indices, col_indices, 1
        )
        # Clamp negative values to avoid numerical errors
        cost_values = torch.clamp(cost_values, min=0.0)
        cost_values = torch.sqrt(cost_values)
        return _make_csr_matrix(
            crow_indices,
            col_indices,
            cost_values,
            (n, n),
            self.device,
        )

    def fit(self, X, Y):
        """

        Parameters
        ----------
        X: (n_samples, n_features) torch.Tensor
            source data
        Y: (n_samples, n_features) torch.Tensor
            target data
        """
        n_features = X.shape[1]
        F = _low_rank_squared_l2(X.T, Y.T)

        init_plan = self._initialize_plan(n_features)
        cost = self._cost(init_plan, F, n_features)

        self.mass = (
            self.sparsity_mask.sum(dim=1)
            .to_dense()
            .to(self.device)
            .type(torch.float64)
        )
        ws = torch.ones_like(self.mass).to(self.device) / self.mass
        wt = ws.clone().to(self.device)

        if self.solver =="sinkhorn_epsilon_scaling":
            _, pi = solver_sinkhorn_eps_scaling_sparse(
                cost=cost,
                ws=ws,
                wt=wt,
                eps=self.reg,
                numItermax=self.max_iter,
                tol=self.tol,
                eval_freq=self.eval_freq,
                stabilization_threshold=1e6,
                verbose=self.verbose,
            )
        elif self.solver == "sinkhorn_stabilized":
            _, pi = solver_sinkhorn_stabilized_sparse(
                cost=cost,
                ws=ws,
                wt=wt,
                eps=self.reg,
                numItermax=self.max_iter,
                tol=self.tol,
                eval_freq=self.eval_freq,
                stabilization_threshold=1e6,
                verbose=self.verbose,
            )
        else:
            raise ValueError(
                f"Solver {self.solver} not recognized. "
                "Please use 'sinkhorn_epsilon_scaling' or 'sinkhorn_stabilized'."
            )

        # Coupling matrix is rescaled by the mass
        self.R = (
            pi.to_sparse_coo().detach()
            @ torch.sparse_coo_tensor(
                torch.stack([torch.arange(n_features)] * 2),
                self.mass,
                (n_features, n_features),
                device=self.device,
            ).to_sparse_coo()
        )

        if self.R.values().isnan().any():
            raise ValueError(
                "Coupling matrix contains NaN values,"
                "try increasing the regularization parameter."
            )

        return self

    def transform(self, X):
        """Transform X using optimal coupling computed during fit.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data to be transformed

        Returns
        -------
        torch.Tensor of shape (n_samples, n_features)
            Transformed data
        """
        return (X @ self.R).to_dense()
