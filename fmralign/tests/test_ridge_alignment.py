import numpy as np
from sklearn.utils.testing import assert_array_almost_equal, assert_greater
from fmralign.alignment_methods import RidgeAlignment
import nibabel
from fmralign.pairwise_alignment import PairwiseAlignment
from fmralign.tests.utils import random_nifti, assert_model_align_better_than_identity


def _test_piecewise_alignment():
    for method in ["RidgeCV", "RRRCV"]:
        print(method)
        pwa = PieceWiseAlignment(
            n_pieces=n_labels, mask=mask_img, standardize=False, detrend=False, method=method)
        pwa.fit(img1, img2)
        img2_ = pwa.transform(img1)
        assert_array_almost_equal(img2_.get_data(), img2.get_data(), 6)


def _test_piecewise_alignment_parallel2():
    for method in ["RidgeCV", "RRRCV"]:
        print(method)
        pwa = PieceWiseAlignment(n_pieces=n_labels, mask=mask_img,
                                 standardize=False, detrend=False, method=method, n_jobs=2)
        pwa.fit(img1, img2)
        img2_ = pwa.transform(img1)
        assert_array_almost_equal(img2_.get_data(), img2.get_data(), 6)


def _test_multiple_piecewise_alignment_shapes():
    for method in ["mean", "RidgeCV", "RRRCV", "permutation"]:
        print(method)
        img1_m = [img1, img1]
        img2_m = [img2, img2]
        pwa = PieceWiseAlignment(
            n_pieces=n_labels, mask=mask_img, standardize=False, detrend=False, method=method)
        pwa.fit(img1_m, img2_m)
        img1_m_ = pwa.transform(img1_m)

        assert img1_m_.get_data().shape == (10, 10, 5, 10)
