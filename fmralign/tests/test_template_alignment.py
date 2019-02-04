import time
import numpy as np
import nibabel
from sklearn.utils.testing import assert_array_almost_equal
from fmralign.template_alignment import TemplateAlignment, euclidian_mean_with_masking
from fmralign.tests.utils import assert_algo_transform_almost_exactly, random_niimg, assert_model_align_better_than_identity


def test_euclidian_mean():
    pass


def test_template_identity():
    img1, mask_img = random_niimg((10, 10, 5, 5))
    img2, _ = random_niimg((10, 10, 5, 5))
    algo = TemplateAlignment(method_alignment='identity')
    pass


def test_template_closer_to_target():
    pass
