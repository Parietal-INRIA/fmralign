# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


def score_voxelwise(ground_truth, prediction, masker, loss,
                    multioutput='raw_values'):
    """
    Calculates loss function for predicted, ground truth
    arrays. Supported scores are R2, correlation, and normalized
    reconstruction error (Bazeille et al., 2019)

    Parameters
    ----------
    ground_truth: 3D or 4D Niimg
        Reference image
    prediction : 3D or 4D Niimg
        Same shape as `ground_truth`
    masker: instance of NiftiMasker or MultiNiftiMasker
        Masker to be used on ground_truth and prediction. For more information see:
        http://nilearn.github.io/manipulating_images/masker_objects.html
    loss : str in ['R2', 'corr', 'n_reconstruction_err']
        The loss function used in scoring. Default is normalized
        reconstruction error.
        'R2' :
            The R2 distance between source and target arrays.
            Best possible score is 1.0 and it can be negative (because the
            model can be arbitrarily worse).
        'corr' :
            The correlation between source and target arrays.
        'n_reconstruction_err' :
            The normalized reconstruction error. A perfect prediction
            yields a value of 1.0
    multioutput: str in ['raw_values', 'uniform_average']
        Defines method for aggregating multiple output scores. Default method
        is 'raw_values' i.e. no aggregation.
        'raw_values' :
            Returns a full set of scores in case of multioutput input.
        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.

    Returns
    -------
    score : float or ndarray of floats
        The score or ndarray of scores if ‘multioutput’ is ‘raw_values’.
        The worst possible score is arbitrarily set to -1 for all metrics.
    """
    X_gt = masker.transform(ground_truth)
    X_pred = masker.transform(prediction)

    if loss is "R2":
        score = r2_score(X_gt, X_pred, multioutput=multioutput)
    elif loss is "n_reconstruction_err":
        score = normalized_reconstruction_error(
            X_gt, X_pred, multioutput=multioutput)
    elif loss is "corr":
        score = np.array([pearsonr(X_gt[:, vox], X_pred[:, vox])[0]  # pearsonr returns both rho and p
                          for vox in range(X_pred.shape[1])])
        if multioutput == "uniform_average":
            score = np.mean(score)
    else:
        raise NameError(
            "Unknown loss. Recognized values are 'R2', 'corr', or 'reconstruction_err'")
    # if the calculated score is less than -1, return -1
    return np.maximum(score, -1)


def normalized_reconstruction_error(y_true, y_pred, sample_weights=None,
                                    multioutput='raw_values'):
    """
    Calculates the normalized reconstruction error
    as defined by Bazeille and colleagues (2019).

    A perfect prediction yields a value of 1.

    Parameters
    ----------
    y_true : arr
        The ground truth array.
    y_pred : arr
        The predicted array.
    sample_weights : arr
        Weights to assign to each sample.
    multioutput: str in ['raw_values', 'uniform_average']
        Defines method for aggregating multiple output scores. Default method
        is 'raw_values' i.e. no aggregation.
        'raw_values' :
            Returns a full set of scores in case of multioutput input.
        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.

    Returns
    -------
    score : float or ndarray of floats
        The score or ndarray of scores if `multioutput` is `raw_values`.

    References
    ----------
    `Bazeille T., Richard H., Janati H., and Thirion B. (2019) Local
    Optimal Transport for Functional Brain Template Estimation.
    In: Chung A., Gee J., Yushkevich P., and Bao S. (eds) Information
    Processing in Medical Imaging. Lecture Notes in Computer Science,
    vol 11492. Springer, Cham.
    DOI: 10.1007/978-3-030-20351-1_18.`
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true) ** 2).sum(axis=0, dtype=np.float64)

    # Include only non-zero values
    nonzero_denominator = (denominator != 0)
    nonzero_numerator = (numerator != 0)
    valid_score = (nonzero_denominator & nonzero_numerator)

    # Calculate reconstruction error
    output_scores = np.ones([y_true.shape[-1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    if multioutput == 'raw_values':
        # return scores individually
        return output_scores

    elif multioutput == 'uniform_average':
        # passing None as weights yields uniform average
        return np.average(output_scores, weights=None)


def reconstruction_ratio(aligned_error, identity_error):
    """
    Calculates the reconstruction error
    as defined by Bazeille and
    colleagues (2019).

    A value greater than 0 indicates that
    voxels are predicted better by aligned data
    than by raw data.

    Parameters
    ----------
    aligned_error : float or ndarray of floats
        The reconstruction error from a given
        functional alignment method
    identity error :  float or ndarray of floats
        The reconstruction error from predicting
        the target subject as the source subject

    References
    ----------
    `Bazeille T., Richard H., Janati H., and Thirion B. (2019) Local
    Optimal Transport for Functional Brain Template Estimation.
    In: Chung A., Gee J., Yushkevich P., and Bao S. (eds) Information
    Processing in Medical Imaging. Lecture Notes in Computer Science,
    vol 11492. Springer, Cham.
    DOI: 10.1007/978-3-030-20351-1_18.`
    """
    num = 1 - aligned_error
    den = 1 - identity_error
    try:
        return 1 - (num / den)
    except ZeroDivisionError:
        return 0.0
