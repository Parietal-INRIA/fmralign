import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


def score_table(loss, X_gt, X_pred, multioutput='raw_values'):
    """
    Calculates loss function for predicted, ground truth
    arrays. Supported scores are R2, correlation, and normalized
    reconstruction error (Bazeille et al., 2019)

    Parameters
    ----------
    loss : str in ['R2', 'corr', 'n_reconstruction_err']
        The loss function used in scoring. Default is normalized
        reconstruction error.
        'R2' :
            The R2 distance between source and target arrays.
        'corr' :
            The correlation between source and target arrays.
        'n_reconstruction_err' :
            The normalized reconstruction error.
    X_gt : arr
        The ground truth array.
    X_pred : arr
        The predicted array
    multioutput: str in [‘raw_values’, ‘uniform_average’]
        Defines aggregating of multiple output scores. Default is raw values.
        ‘raw_values’ :
            Returns a full set of scores in case of multioutput input.
        ‘uniform_average’ :
            Scores of all outputs are averaged with uniform weight.

    Returns
    -------
    score : float or ndarray of floats
        The score or ndarray of scores if ‘multioutput’ is ‘raw_values’.
    """
    if loss is "R2":
        score = r2_score(X_gt, X_pred, multioutput=multioutput)
    elif loss is "n_reconstruction_err":
        score = normalized_reconstruction_error(
            X_gt, X_pred, multioutput=multioutput)
    elif loss is "corr":
        score = np.array([pearsonr(X_gt[:, vox], X_pred[:, vox])[0]  # pearsonr returns both rho and p
                          for vox in range(X_pred.shape[1])])
    else:
        raise NameError(
            "Unknown loss. Recognized values are 'R2', 'corr', or 'reconstruction_err'")
    # if the calculated score is less than -1, return -1
    return np.maximum(score, -1)


def normalized_reconstruction_error(y_true, y_pred, multioutput='raw_values'):
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
    multioutput: str in [‘raw_values’, ‘uniform_average’]
    Defines aggregating of multiple output scores. Default is raw values.
    ‘raw_values’ :
        Returns a full set of scores in case of multioutput input.
    ‘uniform_average’ :
        Scores of all outputs are averaged with uniform weight.
    
    Returns
    -------
    score : float or ndarray of floats
        The score or ndarray of scores if ‘multioutput’ is ‘raw_values’.
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
    output_scores = np.ones([y_true.shape[1]])
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
    aligned_error : float64
        The reconstruction error from a given
        functional alignment method
    identity error :  float64
        The reconstruction error from predicting
        the target subject as the source subject
    """
    num = 1 - aligned_error
    den = 1 - identity_error
    return 1 - (num / den)
