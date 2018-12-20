from sklearn.utils.testing import assert_array_almost_equal, assert_greater
from sklearn.metrics import r2_score
from nilearn.input_data import NiftiMasker
import numpy as np
import nibabel


def assert_class_align_better_than_identity(algo, X, Y):
    """ Tests that the given algorithm align ndarrays X into Y better than identity. This alignment is measured through r2 score.
    """
    print(algo)
    algo.fit(X, Y)
    identity_baseline_score = r2_score(
        Y, X)
    algo_score = r2_score(Y, algo.transform(X))
    assert_greater(algo_score, identity_baseline_score)


def assert_algo_transform_almost_exactly(algo, img1, img2, mask=None):
    """ Tests that the given algorithm manage to transform almost exactly Nifti image img1 into Nifti Image img2
    """
    algo.fit(img1, img2)
    imtest = algo.transform(img1)
    masker = NiftiMasker(mask_img=mask)
    masker.fit()
    assert_array_almost_equal(masker.transform(
        img2), masker.transform(imtest), decimal=6)


def random_nifti(shape):
    """ Produces a random nifti image of shape (shape) and the appropriate mask to use it.
    """
    im = nibabel.Nifti1Image(np.random.random_sample(shape), np.eye(4))
    mask_img = nibabel.Nifti1Image(np.ones(shape[0:3]), np.eye(4))
    return im, mask_img


def assert_model_align_better_than_identity(algo, img1, img2, mask=None):
    """ Tests that the given algorithm align Nifti image img1 into Nifti Image img2 better than identity. This alignment is measured through r2 score.
    """
    algo.fit(img1, img2)
    im_test = algo.transform(img1)
    masker = NiftiMasker(mask)
    masker.fit()
    identity_baseline_score = r2_score(
        masker.transform(img2), masker.transform(img1))
    algo_score = r2_score(masker.transform(img2), masker.transform(
        im_test))
    assert_greater(algo_score, identity_baseline_score)
