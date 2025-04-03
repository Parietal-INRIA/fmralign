# -*- coding: utf-8 -*-

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal
from scipy.linalg import orthogonal_procrustes
from scipy.sparse import csc_matrix

from fmralign.alignment_methods import (
    DiagonalAlignment,
    Identity,
    OptimalTransportAlignment,
    POTAlignment,
    RidgeAlignment,
    ScaledOrthogonalAlignment,
    SparseOT,
    _voxelwise_signal_projection,
    scaled_procrustes,
)
from fmralign.tests.utils import zero_mean_coefficient_determination


def test_scaled_procrustes_algorithmic():
    """Test Scaled procrustes"""
    X = np.random.randn(10, 20)
    Y = np.zeros_like(X)
    R = np.eye(X.shape[1])
    R_test, _ = scaled_procrustes(X, Y)
    assert_array_almost_equal(R, R_test)

    """Test if scaled_procrustes basis is orthogonal"""
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)

    Y = np.random.rand(3, 4)
    Y = Y - Y.mean(axis=1, keepdims=True)

    R, _ = scaled_procrustes(X.T, Y.T)
    assert_array_almost_equal(R.dot(R.T), np.eye(R.shape[0]))
    assert_array_almost_equal(R.T.dot(R), np.eye(R.shape[0]))

    """ Test if it sticks to scipy scaled procrustes in a simple case"""
    X = np.random.rand(4, 4)
    Y = np.random.rand(4, 4)

    R, _ = scaled_procrustes(X, Y)
    R_s, _ = orthogonal_procrustes(Y, X)
    assert_array_almost_equal(R.T, R_s)

    """Test that primal and dual give same results"""
    # number of samples n , number of voxels p
    n, p = 100, 20
    X = np.random.randn(n, p)
    Y = np.random.randn(n, p)
    R1, s1 = scaled_procrustes(X, Y, scaling=True, primal=True)
    R_s, _ = orthogonal_procrustes(Y, X)
    R2, s2 = scaled_procrustes(X, Y, scaling=True, primal=False)
    assert_array_almost_equal(R1, R2)
    assert_array_almost_equal(R2, R_s.T)
    n, p = 20, 100
    X = np.random.randn(n, p)
    Y = np.random.randn(n, p)
    R1, s1 = scaled_procrustes(X, Y, scaling=True, primal=True)
    R_s, _ = orthogonal_procrustes(Y, X)
    R2, s2 = scaled_procrustes(X, Y, scaling=True, primal=False)
    assert_array_almost_equal(s1 * X.dot(R1), s2 * X.dot(R2))


def test_scaled_procrustes_on_simple_exact_cases():
    """Orthogonal Matrix"""
    v = 10
    k = 10
    rnd_matrix = np.random.rand(v, k)
    R, _ = np.linalg.qr(rnd_matrix)
    X = np.random.rand(10, 20)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)
    R_test, _ = scaled_procrustes(X.T, Y.T)
    assert_array_almost_equal(R_test.T, R)

    """Scaled Matrix"""
    X = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 3.0, 4.0, 6.0], [7.0, 8.0, -5.0, -2.0]]
    )

    X = X - X.mean(axis=1, keepdims=True)

    Y = 2 * X
    Y = Y - Y.mean(axis=1, keepdims=True)

    assert_array_almost_equal(
        scaled_procrustes(X.T, Y.T, scaling=True)[0], np.eye(3)
    )
    assert_array_almost_equal(scaled_procrustes(X.T, Y.T, scaling=True)[1], 2)

    """3D Rotation"""
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(1), -np.sin(1)],
            [0.0, np.sin(1), np.cos(1)],
        ]
    )
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)

    R_test, _ = scaled_procrustes(X.T, Y.T)
    assert_array_almost_equal(
        R.dot(np.array([0.0, 1.0, 0.0])), np.array([0.0, np.cos(1), np.sin(1)])
    )
    assert_array_almost_equal(
        R.dot(np.array([0.0, 0.0, 1.0])),
        np.array([0.0, -np.sin(1), np.cos(1)]),
    )
    assert_array_almost_equal(R, R_test.T)

    """Test Scaled_Orthogonal_Alignment on an exact case"""
    ortho_al = ScaledOrthogonalAlignment(scaling=False)
    ortho_al.fit(X.T, Y.T)
    assert_array_almost_equal(ortho_al.transform(X.T), Y.T)


def test_projection_coefficients():
    n_samples = 4
    n_features = 6
    A = np.random.rand(n_samples, n_features)
    C = []
    for i, a in enumerate(A):
        C.append((i + 1) * a)
    c = _voxelwise_signal_projection(A, C, 2)
    assert_array_almost_equal(c, [i + 1 for i in range(n_samples)])


def test_all_classes_R_and_pred_shape_and_better_than_identity():
    """Test all classes on random case"""
    # test on empty data
    X = np.zeros((30, 10))
    for algo in [
        Identity(),
        RidgeAlignment(),
        ScaledOrthogonalAlignment(),
        OptimalTransportAlignment(),
        DiagonalAlignment(),
    ]:
        algo.fit(X, X)
        assert_array_almost_equal(algo.transform(X), X)
    # if trying to learn a fit from array of zeros to zeros (empty parcel)
    # every algo will return a zero matrix
    for n_samples, n_features in [(100, 20), (20, 100)]:
        X = np.random.randn(n_samples, n_features)
        Y = np.random.randn(n_samples, n_features)
        id = Identity()
        id.fit(X, Y)
        identity_baseline_score = zero_mean_coefficient_determination(Y, X)
        assert_array_almost_equal(X, id.transform(X))
        for algo in [
            RidgeAlignment(),
            ScaledOrthogonalAlignment(),
            ScaledOrthogonalAlignment(scaling=False),
            OptimalTransportAlignment(),
            OptimalTransportAlignment(tau=0.995),
            DiagonalAlignment(),
        ]:
            algo.fit(X, Y)
            # test that permutation matrix shape is (20, 20) except for Ridge
            if isinstance(algo.R, csc_matrix):
                R = algo.R.toarray()
                assert R.shape == (n_features, n_features)
            elif not isinstance(algo, RidgeAlignment):
                R = algo.R
                assert R.shape == (n_features, n_features)
            # test pred shape and loss improvement compared to identity
            X_pred = algo.transform(X)
            assert X_pred.shape == X.shape
            algo_score = zero_mean_coefficient_determination(Y, X_pred)
            assert algo_score >= identity_baseline_score


def test_ot_backend():
    n_samples, n_features = 20, 100
    reg = 1e-6  # Test with a small regularization parameter
    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_features)
    pot_algo = POTAlignment(reg=reg)
    sparsity_mask = torch.ones(n_features, n_features).to_sparse_coo()
    torch_algo = SparseOT(sparsity_mask=sparsity_mask, reg=reg)
    pot_algo.fit(X, Y)
    torch_algo.fit(
        torch.tensor(X, dtype=torch.float64),
        torch.tensor(Y, dtype=torch.float64),
    )
    assert_array_almost_equal(
        pot_algo.R, torch_algo.R.to_dense().numpy(), decimal=5
    )


def test_identity_balanced_wasserstein():
    n_samples, n_features = 10, 5
    X = np.random.randn(n_samples, n_features)
    algo = OptimalTransportAlignment(reg=1e-12, tau=1.0)
    algo.fit(X, X)
    # Check if transport matrix P is uniform diagonal
    assert_array_almost_equal(algo.R, np.eye(n_features))
    # Check if transformation preserves input
    assert_array_almost_equal(X, algo.transform(X))


def test_regularization_effect():
    """Test the effect of regularization parameter."""
    n_samples, n_features = 10, 5
    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_features)

    # Compare results with different regularization values
    algo1 = OptimalTransportAlignment(reg=1e-1, tau=1.0)
    algo2 = OptimalTransportAlignment(reg=1e-3, tau=1.0)

    algo1.fit(X, Y)
    algo2.fit(X, Y)

    # Higher regularization should lead to more uniform transport matrix
    assert np.std(algo1.R) < np.std(algo2.R)


def test_tau_effect():
    """Test the effect of tau parameter on mass conservation."""
    n_samples, n_features = 10, 5
    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_features)

    # Compare results with different tau values
    algo1 = OptimalTransportAlignment(reg=1e-3, tau=0.995)
    algo2 = OptimalTransportAlignment(reg=1e-3, tau=0.990)

    algo1.fit(X, Y)
    algo2.fit(X, Y)

    # Lower tau should result in less mass conservation
    assert np.sum(algo1.R.sum(axis=0)) > np.sum(algo2.R.sum(axis=0))
    assert np.sum(algo1.R.sum(axis=1)) > np.sum(algo2.R.sum(axis=1))


def test_sparseot():
    """Test the sparse version of optimal transport."""
    n_samples, n_features = 100, 20
    X = torch.randn(n_samples, n_features, dtype=torch.float64)
    Y = torch.randn(n_samples, n_features, dtype=torch.float64)
    sparsity_mask = torch.ones(n_features, n_features).to_sparse_coo()
    algo = SparseOT(sparsity_mask=sparsity_mask)
    algo.fit(X, Y)
    X_transformed = algo.transform(X)

    assert algo.R.shape == (n_features, n_features)
    assert algo.R.dtype == torch.float64
    assert isinstance(X_transformed, torch.Tensor)
    assert X_transformed.shape == X.shape

    # Test identity transformation
    algo.R = (torch.eye(n_features, dtype=torch.float64)).to_sparse_coo()
    X_transformed = algo.transform(X)
    assert torch.allclose(X_transformed, X)
