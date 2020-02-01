import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from nilearn.input_data import NiftiMasker
import nibabel
import pytest
from fmralign.tests.utils import random_niimg
from fmralign._utils import _make_parcellation, _hierarchical_k_means
import nilearn
from packaging import version
from nilearn._utils.data_gen import generate_fake_fmri


def test_hierarchical_k_means():
    X = [[10, -10, 30], [12, -8, 24]]
    truth_labels = np.tile([0, 1, 2], 5)
    X = np.tile(X, 5).T
    test_labels = _hierarchical_k_means(X, 3)
    truth_labels = np.tile([test_labels[0], test_labels[1], test_labels[2]], 5)
    assert_array_almost_equal(test_labels, truth_labels)


def test_make_parcellation():
    # make_parcellation is built on Nilearn which already has several test for its Parcellation class
    # here we test just the call of the API is right on a simple example
    img, mask_img = random_niimg((7, 6, 8, 5))
    masker = NiftiMasker(mask_img=mask_img).fit()
    n_pieces = 2
    # create a predefined parcellation
    labels_img = nibabel.Nifti1Image(
        np.hstack([np.ones((7, 3, 8)), 2 * np.ones((7, 3, 8))]), np.eye(4))

    methods = ["kmeans", "ward", labels_img, "hierarchical_kmeans"]
    if version.parse(nilearn.__version__) <= version.parse("0.5.2"):
        with pytest.raises(Exception):
            assert make_parcellation(img, "rena", n_pieces, masker)
    else:
        methods.append("rena")
    for clustering_method in methods:
        labels = _make_parcellation(
            img, clustering_method, n_pieces, masker)
        assert(len(np.unique(labels)) == 2)
    # this is an exception on the installed version on nilearn for now ReNA is not released
    # out of developper mode. Once it is ready, you'll be able to call it directly
    # with the latest version of nilearn and this test will evaluate false.
