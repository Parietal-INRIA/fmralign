import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from nilearn.input_data import NiftiMasker
import nibabel
import pytest
from fmralign.tests.utils import random_niimg
from fmralign._utils import _make_parcellation, voxelwise_correlation


def test_make_parcellation():
    # make_parcellation is built on Nilearn which already has several test for its Parcellation class
    # here we test just the call of the API is right on a simple example
    img, mask_img = random_niimg((7, 6, 8, 5))
    masker = NiftiMasker(mask_img=mask_img).fit()
    n_pieces = 2
    # create a predefined parcellation
    labels_img = nibabel.Nifti1Image(
        np.hstack([np.ones((7, 3, 8)), 2 * np.ones((7, 3, 8))]), np.eye(4))
    for clustering_method in ["kmeans", "ward", labels_img]:
        labels = _make_parcellation(
            img, clustering_method, n_pieces, masker)
        assert(len(np.unique(labels)) == 2)

    # this is an exception on the installed version on nilearn for now ReNA is not released
    # out of developper mode. Once it is ready, you'll be able to call it directly
    # with the latest version of nilearn and this test will evaluate false.
    with pytest.raises(Exception):
        assert make_parcellation(img, "rena", n_pieces, masker)

    mask_img.shape


def test_voxelwise_correlation():
    A = np.asarray(
        [[[[1, 1.2, 1, 1.2, 1]], [[1, 1, 1, .2, 1]],  [[1, -1, 1, -1, 1]]]])
    B = np.asarray(
        [[[[0, 0.2, 0, 0.2, 0]], [[.2, 1, 1, 1, 1]], [[-1, 1, -1, 1, -1]]]])
    im_A = nibabel.Nifti1Image(A, np.eye(4))
    im_B = nibabel.Nifti1Image(B, np.eye(4))
    mask_img = nibabel.Nifti1Image(np.ones(im_A.shape[0:3]), np.eye(4))
    masker = NiftiMasker(mask_img=mask_img).fit()
    # check correlation returned is ok
    correlation = voxelwise_correlation(im_A, im_B, masker)
    assert_array_almost_equal(masker.transform(
        correlation)[0], [1., -0.25, -1])
