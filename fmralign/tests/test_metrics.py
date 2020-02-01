import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiMasker
from sklearn.utils.testing import assert_array_almost_equal

from fmralign import metrics

def test_score_voxelwise():
    A = np.asarray([[
        [[1, 1.2, 1, 1.2, 1]],
        [[1, 1, 1, .2, 1]], 
        [[1, -1, 1, -1, 1]]
    ]])
    B = np.asarray([[
        [[0, 0.2, 0, 0.2, 0]],
        [[.2, 1, 1, 1, 1]],
        [[-1, 1, -1, 1, -1]]
    ]])
    im_A = nib.Nifti1Image(A, np.eye(4))
    im_B = nib.Nifti1Image(B, np.eye(4))
    mask_img = nib.Nifti1Image(np.ones(im_A.shape[0:3]), np.eye(4))
    masker = NiftiMasker(mask_img=mask_img).fit()

    # check correlation raw_values
    correlation1 = metrics.score_voxelwise(im_A, im_B,
                                          masker, loss='corr')
    assert_array_almost_equal(correlation1, [1., -0.25, -1])

    # check correlation uniform_average
    correlation2 = metrics.score_voxelwise(im_A, im_B,
                                          masker, loss='corr',
                                          multioutput='uniform_average')
    assert(correlation2.ndim == 0)

    # check R2
    r2 = metrics.score_voxelwise(im_A, im_B, masker, loss='R2')
    assert_array_almost_equal(r2, [-1., -1., -1.])

    # check normalized reconstruction
    norm_rec = metrics.score_voxelwise(im_A, im_B, masker,
                                       loss='n_reconstruction_err')
    assert_array_almost_equal(norm_rec, [0.14966, 0.683168, -1.])


def test_normalized_reconstruction_error():
    A = np.asarray([
        [1, 1.2, 1, 1.2, 1],
        [1, 1, 1, .2, 1],
        [1, -1, 1, -1, 1]
    ])
    B = np.asarray([
        [0, 0.2, 0, 0.2, 0],
        [.2, 1, 1, 1, 1],
        [-1, 1, -1, 1, -1]
    ])
    
    avg_norm_rec = metrics.normalized_reconstruction_error(
        A, B, multioutput='uniform_average')
    np.testing.assert_almost_equal(avg_norm_rec, -0.788203)
