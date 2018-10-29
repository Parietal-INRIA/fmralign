import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from fmralign.alignment_methods import RidgeAlignment


n_samples = 20
n_features = 1000

X = np.random.rand(n_samples, n_features)
Y = np.random.rand(n_samples, n_features)


def test_shapes():
    rh = RidgeAlignment(alpha=0)
    rh.fit(X, Y)
    assert X.shape == (n_samples, n_features)
    assert Y.shape == (n_samples, n_features)
    assert rh.a.shape == (n_features, n_samples)
    assert rh.b.shape == (n_samples, n_features)
    assert rh.transform(X).shape == (n_samples, n_features)


def test_fit_transform():
    rh = RidgeAlignment(alpha=0)
    assert_array_almost_equal(rh.fit_transform(X, Y), Y)


def test_fit_transform_test_set_shape():
    rh = RidgeAlignment(alpha=15).fit(X, Y)
    X_test = np.random.rand(30, n_features)
    assert rh.transform(X_test).shape == X_test.shape
