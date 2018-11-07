import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_greater
from fmralign.alignment_methods import RidgeAlignment


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
    rh.R.score(X, Y)
    assert_array_almost_equal(rh.transform(X), Y, decimal=1)
