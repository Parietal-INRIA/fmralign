"""Some tools to compute correlation matrices. Functions in this module are
meant to be used as a test for the hyperalignment algorithm only."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.manifold import MDS
from scipy.optimize import linear_sum_assignment


#############################################################################
# METRICS
#############################################################################


def compute_correlation(X, Y, metric: str = "correlation"):
    """Compute correlation between X and Y.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The data.
    Y : ndarray of shape (n_samples, n_features)
        The data.
    metric : {'correlation', 'spearman', 'euclidean'}
        The metric to use.

    Returns
    -------
    corr : ndarray of shape (n_samples, n_samples)
        The correlation matrix.
    """
    if metric == "correlation":
        corr = 1 - pairwise_distances(X, Y, metric="correlation")
    elif metric == "spearman":
        corr = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                corr[i, j] = spearmanr(X[i], Y[j])[0]
    elif metric == "euclidean":
        corr = -pairwise_distances(X, Y, metric="euclidean")
    else:
        raise ValueError("Unknown metric")
    return corr


def compute_similarity(X, Y, metric: str = "euclidean"):
    """
    Compute the similarity matrix between two sets of vectors.

    Parameters:
    X (ndarray): First set of vectors.
    Y (ndarray): Second set of vectors.
    metric (str, optional): The distance metric to use. Defaults to "euclidean".

    Returns:
    ndarray: The similarity matrix between X and Y.
    """
    assert X.shape == Y.shape
    n = X.shape[0]

    def vector_sim(u, v):
        if metric == "euclidean":
            return 1 - (np.linalg.norm(u - v) / np.linalg.norm(u + v)) ** 2
        else:
            return (1 - cdist(u, v, metric)).mean()

    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            d = vector_sim(X[i], Y[j])
            sim[i, j] = d
            sim[j, i] = d
    return sim


def compute_pearson_corr(X, Y, linear_assignment: bool = False):
    """Compute Pearson correlation between X and Y.
    X and Y are two lists of matrices of the same shape.
    The returned matrix will be of shape 2N x 2N, where N is the number of matrices in X and Y.
    """

    assert X.shape == Y.shape

    XY = np.concatenate((X, Y), axis=0)
    n = XY.shape[0]
    corr_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr_i_j = pearson_corr_coeff(
                XY[i], XY[j], linear_assignment=linear_assignment
            )
            corr_mat[i, j] = corr_i_j

    return corr_mat


def pearson_corr_coeff(
    M1: np.ndarray,
    M2: np.ndarray,
    absolute: bool = True,
    linear_assignment: bool = True,
):
    """
    Compute Pearson correlation coefficient between matrices M1 and M2.

    Parameters:
        M1 (numpy.ndarray): First matrix.
        M2 (numpy.ndarray): Second matrix.
        absolute (bool, optional): Whether to compute absolute correlation coefficients. Defaults to True.
        linear_assignment (bool, optional): Whether to perform linear assignment optimization. Defaults to True.

    Returns:
        float: Pearson correlation coefficient.
    """
    assert M1.shape == M2.shape

    n = M1.shape[0]
    corr = np.corrcoef(M1, M2)[:n, n:]

    corr = np.abs(corr) if absolute else corr

    if linear_assignment:
        row_ind, col_ind = linear_sum_assignment(corr, maximize=True)

        # permutation of columns and rows
        corr = corr[row_ind, :]
        corr = corr[:, col_ind]

    corr_diag = np.diag(corr)

    corr_coeff = np.mean(corr_diag)

    return corr_coeff


def tuning_correlation(X, Y):
    """Compute pairwise Pearson correlation matrix between two sets of matrices."""
    assert X.shape == Y.shape
    n = X.shape[0]
    corr_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            corr_i_j = pearson_corr_coeff(X[i], Y[j])
            corr_mat[i, j] = corr_i_j
            corr_mat[j, i] = corr_i_j

    return corr_mat


def stimulus_correlation(X, Y, linear_assignment=True, absolute=True):
    """Compute pairwise Pearson correlation matrix between two stimulus matrices."""
    assert X.shape == Y.shape
    n = X.shape[0]
    corr_mat = np.corrcoef(X, Y)[:n, n:]

    if absolute:
        corr_mat = np.abs(corr_mat)

    if linear_assignment:
        row_ind, col_ind = linear_sum_assignment(corr_mat, maximize=True)
        corr_mat = corr_mat[row_ind, :]
        corr_mat = corr_mat[:, col_ind]

    return corr_mat


def matrix_MDS(X, Y, n_components=2, dissimilarity="euclidean"):
    """
    Perform multidimensional scaling (MDS) on the given data matrices X and Y.

    Parameters:
        X (list): The first data matrix.
        Y (list): The second data matrix.
        n_components (int): The number of dimensions in the output space (default is 2).
        dissimilarity (str or array-like): The dissimilarity measure to use. If it is a string other than "precomputed",
                                          the dissimilarity is computed using the Euclidean distance between flattened data points.
                                          If it is "precomputed", the dissimilarity is assumed to be a precomputed dissimilarity matrix.

    Returns:
        tuple: A tuple containing two arrays. The first array represents the transformed data points from matrix X,
               and the second array represents the transformed data points from matrix Y.
    """
    assert len(X) == len(Y)

    if isinstance(dissimilarity, str) and dissimilarity != "precomputed":
        X_flat = [x.flatten() for x in X]
        Y_flat = [y.flatten() for y in Y]
        XY = np.array(X_flat + Y_flat)
        mds = MDS(n_components=n_components, dissimilarity=dissimilarity)
        transformed = mds.fit_transform(XY)

    else:
        mds = MDS(n_components=n_components, dissimilarity="precomputed")
        transformed = mds.fit_transform(dissimilarity)

    return np.array(transformed[: len(X)]), np.array(transformed[len(X) :])


def thread_compute_correlation(X, Y, i, j):
    """
    Compute the correlation between two time series X_i and Y_i.

    Parameters:
    - X (ndarray):
        ndrray of shape (n_samples, n_features) representing the first time series.
    - Y (ndarray):
        ndrray of shape (n_samples, n_features) representing the second time series.
    - i (int):
        Index of the first time series.
    - j (int):
        Index of the second time series.

    Returns:
    diff_TR_corr (ndarray): Array of shape (n_samples * (n_samples - 1),) containing the correlations between different time points.
    same_TR_corr (ndarray): Array of shape (n_samples,) containing the correlations between the same time points.
    empty_diff_TR_corr (ndarray): Empty array.
    empty_same_TR_corr (ndarray): Empty array.
    """
    X_i, Y_i = X[i], Y[i]
    corr = stimulus_correlation(X_i, Y_i, absolute=False)
    same_TR_corr = np.diag(corr)
    # Get all the values except the diagonal in a list
    diff_TR_corr = corr[np.where(~np.eye(corr.shape[0], dtype=bool))]
    diff_TR_corr = diff_TR_corr.flatten()
    if i == j:
        return (
            np.array([]),
            np.array([]),
            [x for x in diff_TR_corr],
            [x for x in same_TR_corr],
        )

    else:
        return diff_TR_corr, same_TR_corr, np.array([]), np.array([])


def multithread_compute_correlation(
    X, Y, absolute=False, linear_assignment=True, n_jobs=40
):
    """
    Compute correlations between pairs of samples in X and Y using multiple threads.

    Args:
        X (ndarray): The first set of samples, with shape (n_samples, n_features).
        Y (ndarray): The second set of samples, with shape (n_samples, n_features).
        absolute (bool, optional): Whether to compute absolute correlations. Defaults to False.
        linear_assignment (bool, optional): Whether to use linear assignment for correlation computation. Defaults to True.
        n_jobs (int, optional): The number of threads to use for parallel computation. Defaults to 40.

    Returns:
        tuple: A tuple containing four arrays:
            - corr_same_sub_diff_TR: Correlations between different time points of the same subject.
            - corr_same_sub_same_TR: Correlations between the same time points of the same subject.
            - corr_diff_sub_diff_TR: Correlations between different time points of different subjects.
            - corr_diff_sub_same_TR: Correlations between the same time points of different subjects.
    """
    from joblib import Parallel, delayed

    def thread_compute_correlation(X, Y, i, j, absolute=False, linear_assignment=True):
        X_i, Y_i = X[i], Y[i]
        corr = stimulus_correlation(X_i, Y_i, absolute=False)
        same_TR_corr = np.diag(corr)
        # Get all the values except the diagonal in a list
        diff_TR_corr = corr[np.where(~np.eye(corr.shape[0], dtype=bool))]
        diff_TR_corr = diff_TR_corr.flatten().astype(np.float16)
        if i == j:
            return (
                np.array([]),
                np.array([]),
                [x for x in diff_TR_corr],
                [x for x in same_TR_corr],
            )

        else:
            return (
                diff_TR_corr.astype(np.float16),
                same_TR_corr.astype(np.float16),
                np.array([]),
                np.array([]),
            )

    from itertools import combinations

    assert X.shape == Y.shape
    n_s = X.shape[0]
    coordinates = list(combinations(range(n_s), 2)) + [(i, i) for i in range(n_s)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(thread_compute_correlation)(X, Y, i, j) for (i, j) in coordinates
    )
    results = list(zip(*results))
    corr_same_sub_diff_TR = results[2]
    corr_same_sub_same_TR = results[3]
    corr_diff_sub_diff_TR = results[0]
    corr_diff_sub_same_TR = results[1]

    corr_same_sub_diff_TR = np.concatenate(corr_same_sub_diff_TR)
    corr_same_sub_same_TR = np.concatenate(corr_same_sub_same_TR)
    corr_diff_sub_diff_TR = np.concatenate(corr_diff_sub_diff_TR)
    corr_diff_sub_same_TR = np.concatenate(corr_diff_sub_same_TR)
    return (
        corr_same_sub_diff_TR,
        corr_same_sub_same_TR,
        corr_diff_sub_diff_TR,
        corr_diff_sub_same_TR,
    )
