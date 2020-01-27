# -*- coding: utf-8 -*-

import pytest
from sklearn.utils.testing import assert_greater
from nilearn.input_data import NiftiMasker
from fmralign.pairwise_alignment import PairwiseAlignment
from fmralign.tests.utils import (assert_algo_transform_almost_exactly,
                                  zero_mean_coefficient_determination,
                                  random_niimg)


def test_unsupported_alignment():
    img1, mask_img = random_niimg((8, 7, 6, 10))
    img2, _ = random_niimg((7, 6, 8, 5))
    args = {'alignment_method': 'scaled_procrustes', 'mask': mask_img}
    algo = PairwiseAlignment(**args)
    with pytest.raises(NotImplementedError):
        algo.fit(img1, img2)


def test_pairwise_identity():
    img1, mask_img = random_niimg((8, 7, 6, 10))

    args_list = [{'alignment_method': 'identity', 'mask': mask_img},
                 {'alignment_method': 'identity', 'n_pieces': 3,
                  'mask': mask_img},
                 {'alignment_method': 'identity', 'n_pieces': 3,
                     'n_bags': 4, 'mask': mask_img},
                 {'alignment_method': 'identity', 'n_pieces': 3,
                     'n_bags': 3, 'mask': mask_img, 'n_jobs': 2}
                 ]
    for args in args_list:
        algo = PairwiseAlignment(**args)
        assert_algo_transform_almost_exactly(
            algo, img1, img1, mask=mask_img)


def test_models_against_identity():
    img1, mask_img = random_niimg((7, 6, 8, 5))
    img2, _ = random_niimg((7, 6, 8, 5))
    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()
    ground_truth = masker.transform(img2)
    identity_baseline_score = zero_mean_coefficient_determination(
        ground_truth, masker.transform(img1))
    for alignment_method in ['permutation',  'ridge_cv', 'scaled_orthogonal',
                             'optimal_transport', 'diagonal']:
        for clustering in ["kmeans", "hierarchical_kmeans"]:
            algo = PairwiseAlignment(
                alignment_method=alignment_method, mask=masker,
                clustering=clustering, n_pieces=2, n_bags=1, n_jobs=1)
            algo.fit(img1, img2)
            im_test = algo.transform(img1)
            algo_score = zero_mean_coefficient_determination(ground_truth,
                                                             masker.transform(
                                                                 im_test))
            assert_greater(algo_score, identity_baseline_score)
