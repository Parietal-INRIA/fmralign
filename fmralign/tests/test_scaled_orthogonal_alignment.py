import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from fmralign.alignment_methods import scaled_procrustes, ScaledOrthogonalAlignment
from scipy.linalg import orthogonal_procrustes


def test_scaled_procrustes_primal_dual():
    n, p = 100, 20
    X = np.random.randn(n, p)
    Y = np.random.randn(n, p)
    R1, s1 = scaled_procrustes(X, Y, scale=True, primal=True)
    R2, s2 = scaled_procrustes(X, Y, scale=True, primal=False)
    assert_array_almost_equal(R1, R2)
    n, p = 20, 100
    X = np.random.randn(n, p)
    Y = np.random.randn(n, p)
    R1, s1 = scaled_procrustes(X, Y, scale=True, primal=True)
    R2, s2 = scaled_procrustes(X, Y, scale=True, primal=False)
    assert_array_almost_equal(R1, R2)


def test_scaled_procrustes_3Drotation():
    R = np.array([[1., 0., 0.], [0., np.cos(1), -np.sin(1)],
                  [0., np.sin(1), np.cos(1)]])
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)

    R_test, _ = scaled_procrustes(X, Y)
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
    R_test, _ = scaled_procrustes(X, Y)
    assert_array_almost_equal(R_test, R)


def test_scaled_procrustes_multiplication():
    X = np.array([[1., 2., 3., 4.],
                  [5., 3., 4., 6.],
                  [7., 8., -5., -2.]])

    X = X - X.mean(axis=1, keepdims=True)

    Y = 2 * X
    Y = Y - Y.mean(axis=1, keepdims=True)

    assert_array_almost_equal(
        scaled_procrustes(X, Y, scale=True)[0], np.eye(3))
    assert_array_almost_equal(scaled_procrustes(X, Y, scale=True)[1], 2)


def test_scaled_procrustes_basis_orthogonal():
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)

    Y = np.random.rand(3, 4)
    Y = Y - Y.mean(axis=1, keepdims=True)

    R, _ = scaled_procrustes(X, Y)
    assert_array_almost_equal(R.dot(R.T), np.eye(R.shape[0]))
    assert_array_almost_equal(R.T.dot(R), np.eye(R.shape[0]))


def test_scaled_procrustes_scipy_orthogonal_procrustes():
    X = np.random.rand(4, 4)
    Y = np.random.rand(4, 4)

    R, _ = scaled_procrustes(X, Y)
    R_s, _ = orthogonal_procrustes(Y, X)
    assert_array_almost_equal(R, R_s)


def test_Scaled_Orthogonal_Alignment_transform():
    X = []
    Y = []

    for i in range(10):
        X_ = np.random.rand(3, 4)
        X.append(X_)
        Y_ = np.random.rand(3, 4)
        Y.append(Y_)

    ortho_al = ScaledOrthogonalAlignment(scale=True)
    ortho_al.fit(X)
    res = ortho_al.transform(Y)
    res_inv = ortho_al.inverse_transform(res)

    for i in range(len(res)):
        assert_array_almost_equal(Y[i], res_inv[i])
