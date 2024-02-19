"""
The linear algebra module. This module contains functions that are often
used in hyperalignment algorithms. Specifically, the robustness of
singular value decomposition (SVD) is enhanced in ``safe_svd`` to avoid
occasional crashes when the operation is performed many times (e.g., in a
searchlight algorithm), and ``svd_pca`` performs PCA based on
``safe_svd``.
"""

import numpy as np
from scipy.linalg import LinAlgError
from scipy.linalg import svd

__all__ = ["safe_svd", "svd_pca", "ridge"]


def safe_svd(X, remove_mean=True):
    """
    Singular value decomposition without occasional LinAlgError crashes.

    The default ``lapack_driver`` of ``scipy.linalg.svd`` is ``'gesdd'``,
    which occassionaly crashes even if the input matrix is not singular.
    This function automatically handles the ``LinAlgError`` when it's
    raised and switches to the ``'gesvd'`` driver in this case.

    The input matrix ``X`` is factorized as ``U @ np.diag(s) @ Vt``.

    Parameters
    ----------
    X : ndarray of shape (M, N)
        The matrix to be decomposed in NumPy array format.
    remove_mean : bool, default=True
        Whether to subtract the mean of each column before the actual SVD
        (True) or not (False). Setting `remove_mean=True` is helpful when
        the SVD is used to perform PCA.

    Returns
    -------
    U : ndarray of shape (M, K)
        Unitary matrix.
    s : ndarray of shape (K,)
        The singular values.
    Vt : ndarray of shape (K, N)
        Unitary matrix.
    """
    if remove_mean:
        X_ = X - X.mean(axis=0, keepdims=True)
    else:
        X_ = X.copy()
    try:
        U, s, Vt = svd(X_, full_matrices=False)
    except LinAlgError:
        U, s, Vt = svd(X_, full_matrices=False, lapack_driver="gesvd")
    del X_
    return U, s, Vt


def svd_pca(X, remove_mean=True):
    """
    Principal component analysis (PCA) based on SVD.

    This function performs a rotation and returns the transformed data in
    PC space. Therefore, its behavior is similar to the ``fit_transform``
    method of ``sklearn.decomposition.PCA``.

    It does not throw away any PCs, and therefore there is no
    dimensionality reduction in the PC space. However, the number of PCs
    might be less than the number of features in ``X``, depending on the
    rank of ``X``.

    Parameters
    ----------
    X : ndarray of shape (M, N)
        The data matrix to be transformed into PC space.
    remove_mean : bool, default=True
        Whether to subtract the mean of each column before the SVD (True)
        or not (False). This parameter should be set to True unless the
        columns already have zero mean.

    Returns
    -------
    X_new : ndarray of shape (M, K)
        The transformed data matrix in PC space.
    """
    U, s, Vt = safe_svd(X, remove_mean=remove_mean)
    X_new = U * s[np.newaxis]
    return X_new


def ridge(X, Y, alpha=10):
    """Solve ridge regression problem for matrix target using SVD.

    Parameters
    ----------
    X : ndarray
        The data matrix of shape (n_samples, n_features).
    Y : ndarray of shape (n_samples, n_targets)
        The target matrix.
    alpha : float
        The regularization parameter.

    Returns
    -------
    betas : ndarray of shape (n_features, n_targets)
        The solution to the ridge regression problem.
    """
    U, s, Vt = safe_svd(X, remove_mean=True)
    d = s / (alpha + s**2)
    d_UT_Y = d[:, np.newaxis] * (U.T @ Y)
    betas = Vt.T @ d_UT_Y
    return betas


def procrustes(X, Y, reflection=True, scaling=False):
    r"""
    The orthogonal Procrustes algorithm.

    Parameters
    ----------
    X : ndarray
        The data matrix to be aligned to Y.
    Y : ndarray
        The "target" data matrix -- the matrix to be aligned to.
    reflection : bool, default=True
        Whether allows reflection in the transformation (True) or not
        (False). Note that even with ``reflection=True``, the solution
        may not contain a reflection if the alignment cannot be improved
        by adding a reflection to the rotation.
    scaling : bool, default=False
        Whether allows global scaling (True) or not (False). Allowing
        scaling can improve alignment quality, but it also changes the
        geometry of data.

    Returns
    -------
    T : ndarray
        The transformation matrix which can be used to align X to Y.
        Depending on the parameters ``reflection`` and ``scaling``, the
        transformation can be a pure rotation, an improper rotation, or a
        pure/improper rotation with global scaling.
    """

    A = Y.T.dot(X).T
    U, s, Vt = safe_svd(A, remove_mean=False)
    T = np.dot(U, Vt)

    if not reflection:
        sign = np.sign(np.linalg.det(T))
        s[-1] *= sign
        if sign < 0:
            T -= np.outer(U[:, -1], Vt[-1, :]) * 2

    if scaling:
        scale = s.sum() / (X.var(axis=0).sum() * X.shape[0])
        T *= scale

    return T
