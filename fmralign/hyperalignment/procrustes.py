import numpy as np
from .linalg import safe_svd


def procrustes(X, Y, reflection=True, scaling=False):
    r"""
    The orthogonal Procrustes algorithm.

    The orthogonal Procrustes algorithm, also known as the classic
    hyperalignment algorithm, is the first hyperalignment algorithm, which
    was introduced in Haxby et al. (2011). It tries to align two
    configurations in a high-dimensional space through an orthogonal
    transformation. The transformation is an improper rotation, which is a
    rotation with an optional reflection. Neither rotation nor reflection
    changes the geometry of the configuration. Therefore, the geometry,
    such as a representational dissimilarity matrix (RDM), remains the
    same in the process.

    Optionally, a global scaling can be added to the algorithm to further
    improve alignment quality. That is, scaling the data with the same
    factor for all directions. Different from rotation and reflection,
    global scaling can change the geometry of the data.

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

    Notes
    -----
    The algorithm tries to minimize the Frobenius norm of the difference
    between transformed data ``X @ T`` and the target ``Y``:

    .. math:: \underset{T}{\arg\min} \lVert XT - Y \rVert_F

    The solution ``T`` differs depending on whether reflection and global
    scaling are allowed. When it's a rotation (pure or improper), it's
    often denoted as ``R``.
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
