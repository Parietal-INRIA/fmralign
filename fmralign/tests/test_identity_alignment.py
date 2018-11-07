import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from fmralign.alignment_methods import Identity
import nibabel
from fmralign.pairwise_alignment import PairwiseAlignment
from fmralign.tests.utils import _test_algo


def test_identity_class():
    X = np.random.randn(10, 20)
    Y = np.random.randn(30, 20)
    id = Identity()
    id.fit(X, Y)
    assert_array_almost_equal(X, id.transform(X))


def test_pairwise_identity():
    img1 = nibabel.Nifti1Image(np.random.rand(10, 10, 5, 5), np.eye(4))
    mask_img = nibabel.Nifti1Image(np.ones((10, 10, 5)), np.eye(4))
    identity = PairwiseAlignment(alignment_method='identity', mask=mask_img)
    _test_algo(identity, img1, img1, mask=mask_img)


def test_piecewise_identity():
    img1 = nibabel.Nifti1Image(np.random.rand(10, 10, 5, 5), np.eye(4))
    mask_img = nibabel.Nifti1Image(np.ones((10, 10, 5)), np.eye(4))
    identity = PairwiseAlignment(
        alignment_method='identity', n_pieces=3, mask=mask_img)
    _test_algo(identity, img1, img1, mask=mask_img)


def test_bagged_piecewise_identity():
    img1 = nibabel.Nifti1Image(np.random.rand(10, 10, 5, 5), np.eye(4))
    mask_img = nibabel.Nifti1Image(np.ones((10, 10, 5)), np.eye(4))
    identity = PairwiseAlignment(
        alignment_method='identity', n_pieces=3, n_bags=10, mask=mask_img)
    _test_algo(identity, img1, img1, mask=mask_img)
