import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from fmralign.alignment_methods import optimal_permutation, Hungarian


def test_hungarian_translation():
    X = np.random.rand(100, 20)
    # translate the data matrix along one axis
    Y = np.roll(X, 10, axis=0)
    hu = Hungarian()
    hu.fit(X, Y)
    X_permutated =
    assert_array_almost_equal(hu.transform(X) Y)

    X = np.random.rand(20, 100)
    Y = np.roll(X, 10, axis=1)
    hu = Hungarian()
    hu.fit(X, Y)
    X_permutated =
    assert_array_almost_equal(hu.transform(X) Y)


def test_optimal_permutation_3Drotation():
    R = np.array([[1., 0., 0.], [0., np.cos(1), -np.sin(1)],
                  [0., np.sin(1), np.cos(1)]])
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)

    R_test = optimal_permutation(X, Y)
    assert_array_almost_equal(
        R.dot(np.array([0., 1., 0.])),
        np.array([0., np.cos(1), np.sin(1)])
    )
    assert_array_almost_equal(
        R.dot(np.array([0., 0., 1.])),
        np.array([0., -np.sin(1), np.cos(1)])
    )
    assert_array_almost_equal(R, R_test)
