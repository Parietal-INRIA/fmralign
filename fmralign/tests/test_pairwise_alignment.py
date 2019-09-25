# -*- coding: utf-8 -*-

import numpy as np
import nibabel
from sklearn.utils.testing import assert_array_almost_equal, assert_greater
from nilearn.input_data import NiftiMasker
from fmralign.pairwise_alignment import PairwiseAlignment
from fmralign.tests.utils import assert_algo_transform_almost_exactly, \
    random_niimg, assert_model_align_better_than_identity, \
    zero_mean_coefficient_determination
from fmralign.alignment_methods import optimal_permutation, Hungarian


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
        algo = PairwiseAlignment(
            alignment_method=alignment_method, mask=mask_img,
            n_pieces=2, n_bags=1, n_jobs=1)
        algo.fit(img1, img2)
        im_test = algo.transform(img1)
        algo_score = zero_mean_coefficient_determination(ground_truth,
                                                         masker.transform(
                                                             im_test))
        assert_greater(algo_score, identity_baseline_score)
