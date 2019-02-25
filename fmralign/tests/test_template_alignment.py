import time
import numpy as np
import nibabel
from sklearn.utils.testing import assert_array_almost_equal
from fmralign.template_alignment import TemplateAlignment, euclidian_mean_with_masking
from fmralign.tests.utils import assert_algo_transform_almost_exactly, random_niimg, assert_model_align_better_than_identity


import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
import nibabel
from nilearn.input_data import NiftiMasker
from functional_alignment.template_alignment import euclidian_mean


def test_euclidian_mean():

    sub = np.tile(np.ones((5, 4)), (5, 5, 1, 1))

    sub_1 = nibabel.Nifti1Image(sub, np.eye(4))
    sub_2 = nibabel.Nifti1Image(2 * sub, np.eye(4))
    sub_3 = nibabel.Nifti1Image(3 * sub, np.eye(4))

    ref_template = 2 * sub

    mask_img = nibabel.Nifti1Image(np.ones((5, 5, 5)), np.eye(4))

    imgs = [sub_1, sub_2, sub_3]
    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()
    # apply class
    euclidian_template = euclidian_mean(imgs, masker)
    assert_array_almost_equal(
        ref_template, euclidian_template.get_data())


def test_euclidian_template_wt_masking():

    pass


def test_template_identity():
    pass


def test_template_closer_to_target():
    pass
