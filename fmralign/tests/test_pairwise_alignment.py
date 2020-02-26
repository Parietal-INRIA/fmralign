# -*- coding: utf-8 -*-
import copy
import pytest
import numpy as np
from sklearn.utils.testing import assert_greater
from nilearn.input_data import NiftiMasker
from nilearn.image import new_img_like
from fmralign.pairwise_alignment import PairwiseAlignment, fit_one_piece
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


def test_optimal_transport_error_big_parcels():
    n_voxels, n_features = 6000, 10
    X, Y = np.ones((n_features, n_voxels)), np.ones((n_features, n_voxels))
    with pytest.warns(UserWarning):
        estimator = fit_one_piece(X, Y, "optimal_transport")


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

    # test intersection of clustering and mask
    data_mask = copy.deepcopy(mask_img.get_fdata())
    data_mask[0] = 0
    # create ground truth
    clustering_mask = new_img_like(mask_img, data_mask)
    data_clust = copy.deepcopy(data_mask)
    data_clust[1] = 3
    # create 2-parcels clustering, smaller than background
    clustering = new_img_like(mask_img, data_clust)

    # clustering is smaller than mask
    assert (mask_img.get_fdata() > 0).sum() > (clustering.get_data() > 0).sum()
    algo = PairwiseAlignment(alignment_method='identity',
                             mask=mask_img, clustering=clustering)
    with pytest.warns(UserWarning):
        algo.fit(img1, img1)
    assert (algo.mask.get_fdata() > 0).sum() == (
        clustering.get_fdata() > 0).sum()

    # test warning raised if parcel is 0 :
    null_im = new_img_like(img1, np.zeros_like(img1.get_fdata()))
    with pytest.warns(UserWarning):
        algo.fit(null_im, null_im)


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
