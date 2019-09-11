from fmralign._utils import hierarchical_k_means, voxelwise_correlation
import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from nilearn.input_data import NiftiMasker
import nibabel


def test_hierarchical_k_means():
    X = [[10, -10, 30], [12, -8, 24]]
    truth_labels = np.tile([0, 1, 2], 5)
    X = np.tile(X, 5).T

    test_labels = hierarchical_k_means(X, 3)
    truth_labels = np.tile([test_labels[0], test_labels[1], test_labels[2]], 5)
    assert_array_almost_equal(test_labels, truth_labels)


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
