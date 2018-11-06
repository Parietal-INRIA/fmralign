ccdimport numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.testing import assert_array_almost_equal
import nibabel
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_roi

from functional_alignment.piecewise_alignment import PieceWiseAlignment, create_labels
from functional_alignment.rrr_cv import RRRCV
from functional_alignment.ridge_cv import RidgeCV

img1 = np.random.rand(10, 10, 5, 5)
img1 = nibabel.Nifti1Image(img1, np.eye(4))

img2 = np.random.rand(10, 10, 5, 5)
img2 = nibabel.Nifti1Image(img2, np.eye(4))

mask_img = nibabel.Nifti1Image(np.ones((10, 10, 5)), np.eye(4))

n_labels = 10


def test_hungarian_pairwise_alignment():
    X = np.array([[1., 4., 10], [1.5, 5, 10], [1, 5, 11], [1, 5.5, 8]]).T
    img1_3d = nibabel.Nifti1Image(X, np.eye(4))
    img1_4d = nibabel.Nifti1Image([X, X], np.eye(4))
    # translate the data matrix along one axis
    Y = np.roll(X, 2, axis=0)
    img2_3d = nibabel.Nifti1Image(Y, np.eye(4))
    img2_4d = nibabel.Nifti1Image([Y, Y], np.eye(4))

    mask_img = nibabel.Nifti1Image(np.ones(X.shape), np.eye(4))

    opt = optimal_permutation(X, Y).toarray()
    assert_array_almost_equal(opt.dot(X), Y)

    hu = Hungarian()
    hu.fit(X, Y)
    assert_array_almost_equal(hu.transform(X), Y)


def test_mean_piecewise_alignment():
    pwa = PieceWiseAlignment(n_pieces=n_labels, mask=mask_img,
                             standardize=False, detrend=False, method="mean")
    pwa.fit(img1, img2)
    img1_ = pwa.transform(img1)
    assert_array_almost_equal(img1.get_data(), img1_.get_data())


def test_piecewise_alignment():
    for method in ["RidgeCV", "RRRCV"]:
        print(method)
        pwa = PieceWiseAlignment(
            n_pieces=n_labels, mask=mask_img, standardize=False, detrend=False, method=method)
        pwa.fit(img1, img2)
        img2_ = pwa.transform(img1)
        assert_array_almost_equal(img2_.get_data(), img2.get_data(), 6)


def test_piecewise_alignment_parallel2():
    for method in ["RidgeCV", "RRRCV"]:
        print(method)
        pwa = PieceWiseAlignment(n_pieces=n_labels, mask=mask_img,
                                 standardize=False, detrend=False, method=method, n_jobs=2)
        pwa.fit(img1, img2)
        img2_ = pwa.transform(img1)
        assert_array_almost_equal(img2_.get_data(), img2.get_data(), 6)


def test_multiple_piecewise_alignment_shapes():
    for method in ["mean", "RidgeCV", "RRRCV", "permutation"]:
        print(method)
        img1_m = [img1, img1]
        img2_m = [img2, img2]
        pwa = PieceWiseAlignment(
            n_pieces=n_labels, mask=mask_img, standardize=False, detrend=False, method=method)
        pwa.fit(img1_m, img2_m)
        img1_m_ = pwa.transform(img1_m)

        assert img1_m_.get_data().shape == (10, 10, 5, 10)
