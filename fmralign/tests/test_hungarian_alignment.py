import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from fmralign.alignment_methods import optimal_permutation, Hungarian


def test_hungarian_translation():
    X = np.array([[1., 4., 10], [1.5, 5, 10], [1, 5, 11], [1, 5.5, 8]])

    # translate the data matrix along features axis (voxels are permutated)
    Y = np.roll(X, 2, axis=1)

    opt = optimal_permutation(X, Y).toarray()
    assert_array_almost_equal(opt.dot(X.T).T, Y)

    hu = Hungarian()
    hu.fit(X, Y)
    assert_array_almost_equal(hu.transform(X), Y)

    U = np.vstack([X.T, 2 * X.T])
    V = np.roll(U, 4, axis=1)

    opt = optimal_permutation(U, V).toarray()
    assert_array_almost_equal(opt.dot(U.T).T, V)

    hu = Hungarian()
    hu.fit(U, V)
    assert_array_almost_equal(hu.transform(U), V)
