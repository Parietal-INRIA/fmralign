import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from fmralign.alignment_methods import RidgeAlignment


n_samples = 40
n_features = 100

X = np.random.rand(n_samples, n_features)
Y = np.random.rand(n_samples, n_features)


def test_parameters():
    test_alphas = [1, 2, 3]
    test_cv = 6
    rh = RidgeAlignment(alphas=test_alphas, cv=test_cv)
    assert(rh.alphas == test_alphas)
    assert(rh.cv == test_cv)


def test_shapes():
    rh = RidgeAlignment()
    rh.alphas
    rh.fit(X, Y)
    assert X.shape == (n_samples, n_features)
    assert Y.shape == (n_samples, n_features)
    assert rh.transform(X).shape == (n_samples, n_features)
    assert_array_almost_equal(rh.transform(X), Y)
