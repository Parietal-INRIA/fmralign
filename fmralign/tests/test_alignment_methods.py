import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from fmralign.alignment_methods import Identity, optimal_permutation, Hungarian
from fmralign.pairwise_alignment import PairwiseAlignment
from fmralign.tests.utils import assert_algo_transform_almost_exactly, random_nifti


# Test Scaled procrustes

import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from fmralign.alignment_methods import scaled_procrustes, ScaledOrthogonalAlignment
from scipy.linalg import orthogonal_procrustes


def test_procrustes_null_input():
    X = np.random.randn(10, 20)
    Y = np.zeros_like(X)
    R = np.eye(X.shape[1])
    R_test, _ = scaled_procrustes(X, Y)
    assert_array_almost_equal(R, R_test.toarray())


def test_scaled_procrustes_primal_dual():
    n, p = 100, 20
    X = np.random.randn(n, p)
    Y = np.random.randn(n, p)
    R1, s1 = scaled_procrustes(X, Y, scaling=True, primal=True)
    R2, s2 = scaled_procrustes(X, Y, scaling=True, primal=False)
    assert_array_almost_equal(R1, R2)
    n, p = 20, 100
    X = np.random.randn(n, p)
    Y = np.random.randn(n, p)
    R1, s1 = scaled_procrustes(X, Y, scaling=True, primal=True)
    R2, s2 = scaled_procrustes(X, Y, scaling=True, primal=False)
    assert_array_almost_equal(R1.dot(X.T), R2.dot(X.T))


def test_scaled_procrustes_3Drotation():
    R = np.array([[1., 0., 0.], [0., np.cos(1), -np.sin(1)],
                  [0., np.sin(1), np.cos(1)]])
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)

    R_test, _ = scaled_procrustes(X.T, Y.T)
    assert_array_almost_equal(
        R.dot(np.array([0., 1., 0.])),
        np.array([0., np.cos(1), np.sin(1)])
    )
    assert_array_almost_equal(
        R.dot(np.array([0., 0., 1.])),
        np.array([0., -np.sin(1), np.cos(1)])
    )
    assert_array_almost_equal(R, R_test)


def test_scaled_procrustes_orthogonalmatrix():
    v = 10
    k = 10
    rnd_matrix = np.random.rand(v, k)
    R, _ = np.linalg.qr(rnd_matrix)
    X = np.random.rand(10, 20)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)
    R_test, _ = scaled_procrustes(X.T, Y.T)
    assert_array_almost_equal(R_test, R)


def test_scaled_procrustes_multiplication():
    X = np.array([[1., 2., 3., 4.],
                  [5., 3., 4., 6.],
                  [7., 8., -5., -2.]])

    X = X - X.mean(axis=1, keepdims=True)

    Y = 2 * X
    Y = Y - Y.mean(axis=1, keepdims=True)

    assert_array_almost_equal(
        scaled_procrustes(X.T, Y.T, scaling=True)[0], np.eye(3))
    assert_array_almost_equal(scaled_procrustes(X.T, Y.T, scaling=True)[1], 2)


def test_scaled_procrustes_basis_orthogonal():
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)

    Y = np.random.rand(3, 4)
    Y = Y - Y.mean(axis=1, keepdims=True)

    R, _ = scaled_procrustes(X.T, Y.T)
    assert_array_almost_equal(R.dot(R.T), np.eye(R.shape[0]))
    assert_array_almost_equal(R.T.dot(R), np.eye(R.shape[0]))


def test_scaled_procrustes_scipy_orthogonal_procrustes():
    X = np.random.rand(4, 4)
    Y = np.random.rand(4, 4)

    R, _ = scaled_procrustes(X, Y)
    R_s, _ = orthogonal_procrustes(Y, X)
    assert_array_almost_equal(R, R_s)


def test_hungarian_translation():
    X = np.array([[1., 4., 10], [1.5, 5, 10], [1, 5, 11], [1, 5.5, 8]])

    # translate the data matrix along features axis (voxels are permutated)
    Y = np.roll(X, 2, axis=1)

    opt = optimal_permutation(X, Y).toarray()
    assert_array_almost_equal(opt.dot(X.T).T, Y)

    U = np.vstack([X.T, 2 * X.T])
    V = np.roll(U, 4, axis=1)

    opt = optimal_permutation(U, V).toarray()
    assert_array_almost_equal(opt.dot(U.T).T, V)


def test_identity_class():
    X = np.random.randn(10, 20)
    Y = np.random.randn(30, 20)
    id = Identity()
    id.fit(X, Y)
    assert_array_almost_equal(X, id.transform(X))


def test_Scaled_Orthogonal_Alignment_3Drotation():
    R = np.array([[1., 0., 0.], [0., np.cos(1), -np.sin(1)],
                  [0., np.sin(1), np.cos(1)]])
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)

    ortho_al = ScaledOrthogonalAlignment(scaling=False)
    ortho_al.fit(X.T, Y.T)

    assert_array_almost_equal(
        ortho_al.transform(X),
        Y)


def test_parameters():
    test_alphas = [1, 2, 3]
    test_cv = 6
    rh = RidgeAlignment(alphas=test_alphas, gcv=test_cv)
    assert(rh.alphas == test_alphas)
    assert(rh.gcv == test_cv)


def test_RidgeAlignment():
    n_samples, n_features = 6, 6
    Y = np.random.randn(n_samples // 2, n_features)
    Y = np.concatenate((Y, Y))
    X = np.random.randn(n_samples // 2, n_features)
    X = np.concatenate((X, X), axis=0)
    rh = RidgeAlignment()
    rh.alphas
    rh.fit(X, Y)
    assert_greater(rh.R.score(X, Y), 0.9)
    assert_array_almost_equal(rh.transform(X), Y, decimal=1)
