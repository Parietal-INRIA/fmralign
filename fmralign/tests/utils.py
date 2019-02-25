from sklearn.utils.testing import assert_array_almost_equal, assert_greater
from sklearn.metrics import r2_score
from nilearn.input_data import NiftiMasker
import numpy as np
import nibabel


def assert_class_align_better_than_identity(algo, X, Y):
    """ Tests that the given algorithm align ndarrays X into Y better than \
    identity. This alignment is measured through r2 score.
    """
    print(algo)
    algo.fit(X, Y)
    identity_baseline_score = zero_mean_coefficient_determination(
        Y, X)
    algo_score = zero_mean_coefficient_determination(Y, algo.transform(X))
    assert_greater(algo_score, identity_baseline_score)


def assert_algo_transform_almost_exactly(algo, img1, img2, mask=None):
    """ Tests that the given algorithm manage to transform almost exactly Nifti\
     image img1 into Nifti Image img2
    """
    algo.fit(img1, img2)
    imtest = algo.transform(img1)
    masker = NiftiMasker(mask_img=mask)
    masker.fit()
    assert_array_almost_equal(masker.transform(
        img2), masker.transform(imtest), decimal=6)


def random_niimg(shape):
    """ Produces a random nifti image of shape (shape) and the appropriate \
    mask to use it.
    """
    im = nibabel.Nifti1Image(np.random.random_sample(shape), np.eye(4))
    mask_img = nibabel.Nifti1Image(np.ones(shape[0:3]), np.eye(4))
    return im, mask_img


def assert_model_align_better_than_identity(algo, img1, img2, mask=None):
    """ Tests that the given algorithm align Nifti image img1 into Nifti \
    Image img2 better than identity. Proficiency is measured through r2 score.
    """
    algo.fit(img1, img2)
    im_test = algo.transform(img1)
    masker = NiftiMasker(mask)
    masker.fit()
    identity_baseline_score = zero_mean_coefficient_determination(
        masker.transform(img2), masker.transform(img1))
    algo_score = zero_mean_coefficient_determination(
        masker.transform(img2), masker.transform(im_test))
    assert_greater(algo_score, identity_baseline_score)


def zero_mean_coefficient_determination(y_true, y_pred, sample_weight=None,
                                        multioutput="uniform_average"):
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if sample_weight is not None:
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = (weight * (y_true) ** 2).sum(axis=0, dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0

    if multioutput == 'raw_values':
        # return scores individually
        return output_scores
    elif multioutput == 'uniform_average':
        # passing None as weights results is uniform mean
        avg_weights = None
    elif multioutput == 'variance_weighted':
        avg_weights = (weight * (y_true - np.average(y_true, axis=0,
                                                     weights=sample_weight))
                       ** 2).sum(axis=0, dtype=np.float64)
        # avoid fail on constant y or one-element arrays
        if not np.any(nonzero_denominator):
            if not np.any(nonzero_numerator):
                return 1.0
            else:
                return 0.0
    else:
        avg_weights = multioutput

    return np.average(output_scores, weights=avg_weights)
