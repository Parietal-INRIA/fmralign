# -*- coding: utf-8 -*-

import time
import numpy as np
import nibabel
from sklearn.utils.testing import assert_array_almost_equal
from fmralign.template_alignment import TemplateAlignment, euclidian_mean
from fmralign.tests.utils import assert_algo_transform_almost_exactly, random_niimg, assert_model_align_better_than_identity
from nilearn.image import index_img


import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
import nibabel
from nilearn.input_data import NiftiMasker
from fmralign.template_alignment import euclidian_mean

# Could change np.ones by np.rand normally
sub = np.tile(np.ones((5, 4)), (5, 5, 1, 1))

sub_1 = nibabel.Nifti1Image(sub, np.eye(4))
sub_2 = nibabel.Nifti1Image(2 * sub, np.eye(4))
sub_3 = nibabel.Nifti1Image(3 * sub, np.eye(4))

ref_template = 2 * sub

mask_img = nibabel.Nifti1Image(np.ones((5, 5, 5)), np.eye(4))

imgs = [sub_1, sub_2, sub_3]
masker = NiftiMasker(mask_img=mask_img)
masker.fit()


def test_euclidian_mean():

    # apply class
    euclidian_template = euclidian_mean(imgs, masker)
    assert_array_almost_equal(
        ref_template, euclidian_template.get_data())


def test_template_identity():

    args_list = [{'alignment_method': 'identity', 'mask': mask_img},
                 {'alignment_method': 'identity', 'n_pieces': 3, 'mask': mask_img},
                 {'alignment_method': 'identity', 'n_pieces': 3,
                     'n_bags': 10, 'mask': mask_img},
                 {'alignment_method': 'identity', 'n_pieces': 3,
                     'n_bags': 10, 'mask': mask_img, 'n_jobs': 2}
                 ]
    for args in args_list:
        algo = TemplateAlignment(**args)
        # Learning a template which is
        algo.fit(imgs)
        # test template
        assert_array_almost_equal(
            ref_template, algo.template.get_data())

        predicted_imgs = algo.transform(
            sub_1, train_index=range(3), test_index=range(3, 5))
        assert_array_almost_equal(index_img(sub_1, range(3, 5)
                                            ).get_data(), predicted_imgs.get_data())

    # Try to input some list, check that the template is two time the reference,

# THEN INPUT SOME LIST OF  ONES AND SEE IF THE TRANSFORM WILL CORRECTLY LEARN TO MULTIPLY BY 2 THE TEST images
# APART FROM THAT


def test_template_closer_to_target():
    pass
# Take two random images, test that their template is closer to the mean than to any of the three for every method
