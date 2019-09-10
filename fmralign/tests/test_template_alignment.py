# -*- coding: utf-8 -*-
import pytest
from sklearn.utils.testing import assert_array_almost_equal, assert_greater

from nilearn.input_data import NiftiMasker
from nilearn.image import math_img, concat_imgs, index_img

from fmralign.template_alignment import TemplateAlignment, _rescaled_euclidean_mean
from fmralign.tests.utils import random_niimg, zero_mean_coefficient_determination


def test_template_identity():

    n = 10
    im, mask_img = random_niimg((6, 5, 3))
    sub_1 = concat_imgs(n * [im])
    sub_2 = math_img("2 * img", img=sub_1)
    sub_3 = math_img("3 * img", img=sub_1)

    ref_template = sub_2
    subs = [sub_1, sub_2, sub_3]
    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()

    # test euclidian mean function
    euclidian_template = _rescaled_euclidean_mean(subs, masker)
    assert_array_almost_equal(
        ref_template.get_data(), euclidian_template.get_data())

    args_list = [{'alignment_method': 'identity', 'mask': masker},
                 {'alignment_method': 'identity', 'mask': masker, 'n_jobs': 2},
                 {'alignment_method': 'identity', 'n_pieces': 3, 'mask': masker},
                 {'alignment_method': 'identity', 'n_pieces': 3,
                     'n_bags': 2, 'mask': masker}
                 ]
    for args in args_list:
        algo = TemplateAlignment(**args)
        # Learning a template which is
        algo.fit(subs)
        # test template
        assert_array_almost_equal(
            ref_template.get_data(), algo.template.get_data())
        predicted_imgs = algo.transform(
            [index_img(sub_1, range(8))], train_index=range(8),
            test_index=range(8, 10))
        ground_truth = index_img(ref_template, range(8, 10))
        assert_array_almost_equal(
            ground_truth.get_data(), predicted_imgs[0].get_data())

    # test last algo transform with wrong indexes
    train_inds, test_inds = [[0, 1], [1, 3, 10],
                             [4, 11]], [[6, 8, 29], [4, 6], [4, 11]]
    for train_ind, test_ind in zip(train_inds, test_inds):
        with pytest.raises(Exception):
            assert algo.transform(
                [index_img(sub_1, range(8))], train_index=train_ind, test_index=test_ind)


def test_template_closer_to_target():
    n_samples = 6

    subject_1, mask_img = random_niimg((6, 5, 3, n_samples))
    subject_2, _ = random_niimg((6, 5, 3, n_samples))

    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()

    # Calculate metric between each subject and the average

    sub_1 = masker.transform(subject_1)
    sub_2 = masker.transform(subject_2)
    subs = [subject_1, subject_2]
    average_img = _rescaled_euclidean_mean(subs, masker)
    avg_data = masker.transform(average_img)
    mean_distance_1 = zero_mean_coefficient_determination(sub_1, avg_data)
    mean_distance_2 = zero_mean_coefficient_determination(sub_2, avg_data)

    for alignment_method in ['permutation',  'ridge_cv', 'scaled_orthogonal',
                             'optimal_transport', 'diagonal']:
        algo = TemplateAlignment(
            alignment_method=alignment_method, n_pieces=3, n_bags=2, mask=masker)
        # Learn template
        algo.fit(subs)
        # Assess template is closer to mean than both images
        template_data = masker.transform(algo.template)
        template_mean_distance = zero_mean_coefficient_determination(
            avg_data, template_data)
        assert_greater(template_mean_distance, mean_distance_1)
        assert_greater(template_mean_distance, mean_distance_2)
