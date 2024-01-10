""" Local template computation functions. Those functions are part of the warp hyperalignment
introducted by Feilong Ma et al. 2023. The functions are adapted from the original code and
adapted for more general regionations approches.
"""


import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd

from .linalg import safe_svd
from .linalg import procrustes


def PCA_decomposition(
    X, n_components=None, flavor="sklearn", adjust_ns=False, demean=True
):
    """Decompose concatenated data matrices using PCA/SVD.

    Parameters
    ----------
    X : ndarray of shape (ns, nt, nv)
        The input data array.
    n_components : int or None
        The number of components to keep. If None, all components are kept.
    flavor : {'sklearn', 'svd'}
        Wethter to use sklearn or the custom SVD implementation.
    adjust_ns : bool
        Whether to adjust the variance of the output so that it doesn't increase with the number of subjects.
    demean : bool
        Whether to remove the mean of the columns prior to SVD.

    Returns
    -------
    XX : ndarray of shape (nt, npc)
    cc : ndarray of shape (npc, ns, nv)
    """
    ns, nt, nv = X.shape
    X = X.transpose(1, 0, 2).reshape(nt, ns * nv).astype(np.float32)
    if flavor == "sklearn":
        try:
            if demean:
                pca = PCA(n_components=n_components, random_state=0)
                XX = pca.fit_transform(X)
                cc = pca.components_.reshape(-1, ns, nv)
                if adjust_ns:
                    XX /= np.sqrt(ns)
                return XX.astype(np.float32), cc
            else:
                U, s, Vt = randomized_svd(
                    X,
                    (n_components if n_components is not None else min(X.shape)),
                    random_state=0,
                )
                if adjust_ns:
                    XX = U[:, :n_components] * (
                        s[np.newaxis, :n_components] / np.sqrt(ns)
                    )
                else:
                    XX = U[:, :n_components] * (s[np.newaxis, :n_components])
                cc = Vt[:n_components].reshape(-1, ns, nv)
                return XX.astype(np.float32), cc
        except:  # noqa: E722
            return PCA_decomposition(
                X,
                n_components=n_components,
                flavor="svd",
                adjust_ns=adjust_ns,
                demean=demean,
            )
    elif flavor == "svd":
        U, s, Vt = safe_svd(X)
        if adjust_ns:
            XX = U[:, :n_components] * (s[np.newaxis, :n_components] / np.sqrt(ns))
        else:
            XX = U[:, :n_components] * (s[np.newaxis, :n_components])
        cc = Vt[:n_components].reshape(-1, ns, nv)
        return XX.astype(np.float32), cc
    else:
        raise NotImplementedError


def compute_PCA_template(X, sl=None, max_npc=None, flavor="sklearn", demean=False):
    """
    Compute the PCA template from the input data.

    Parameters:
    -----------

    - X (ndarray):
        The input data array of shape (n_samples, n_features, n_timepoints).
    - sl (slice, optional):
        The slice of timepoints to consider. Defaults to None.
    - max_npc (int, optional):
        The maximum number of principal components to keep. Defaults to None.
    - flavor (str, optional):
        The flavor of PCA algorithm to use. Defaults to "sklearn".
    - demean (bool, optional):
        Whether to demean the data before performing PCA. Defaults to False.

    Returns:
    --------

    XX (ndarray):
        The PCA template array of shape (n_samples, n_features, n_components).
    """
    if sl is not None:
        X = X[:, :, sl]
    else:
        X = X
    max_npc = min(X.shape[1], X.shape[2])
    XX, cc = PCA_decomposition(
        X, n_components=max_npc, flavor=flavor, adjust_ns=True, demean=demean
    )
    return XX.astype(np.float32)


def compute_procrustes_template(
    X,
    region=None,
    reflection=True,
    scaling=False,
    zscore_common=True,
    level2_iter=1,
    X2=None,
    debug=False,
):
    """
    Compute the Procrustes template for a given set of data.

    Args:
        X (ndarray): The input data array of shape (n_samples, n_features, n_regions).
        region (int or None, optional): The index of the region to consider. If None, all regions are considered. Defaults to None.
        reflection (bool, optional): Whether to allow reflection in the Procrustes alignment. Defaults to True.
        scaling (bool, optional): Whether to allow scaling in the Procrustes alignment. Defaults to False.
        zscore_common (bool, optional): Whether to z-score the aligned data to have zero mean and unit variance. Defaults to True.
        level2_iter (int, optional): The number of iterations for the level 2 alignment. Defaults to 1.
        X2 (ndarray or None, optional): The second set of input data array of shape (n_samples, n_features, n_regions). Only used for level 2 alignment. Defaults to None.
        debug (bool, optional): Whether to display progress bars during alignment. Defaults to False.

    Returns:
        ndarray: The computed Procrustes template.

    """
    if region is not None:
        X = X[:, :, region]
    common_space = np.copy(X[0])
    aligned_X = [X[0]]
    iter_X = X[1:]
    for x in iter_X:
        T = procrustes(x, common_space, reflection=reflection, scaling=scaling)
        aligned_x = x.dot(T)
        if zscore_common:
            aligned_x = np.nan_to_num(zscore(aligned_x, axis=0))
        aligned_X.append(aligned_x)
        common_space = (common_space + aligned_x) * 0.5
        if zscore_common:
            common_space = np.nan_to_num(zscore(common_space, axis=0))

    aligned_X2 = []
    iter2 = range(level2_iter)

    for level2 in iter2:
        common_space = np.zeros_like(X[0])
        for x in aligned_X:
            common_space += x
        for i, x in enumerate(X):
            reference = (common_space - aligned_X[i]) / float(len(X) - 1)
            if zscore_common:
                reference = np.nan_to_num(zscore(reference, axis=0))
            T = procrustes(x, reference, reflection=reflection, scaling=scaling)
            if level2 == level2_iter - 1 and X2 is not None:
                aligned_X2.append(X2[i].dot(T))
            aligned_X[i] = x.dot(T)

    common_space = np.sum(aligned_X, axis=0)
    if zscore_common:
        common_space = np.nan_to_num(zscore(common_space, axis=0))
    else:
        common_space /= float(len(X))
    if X2 is not None:
        common_space2 = np.zeros_like(X2[0])
        for x in aligned_X2:
            common_space2 += x
        if zscore_common:
            common_space2 = np.nan_to_num(zscore(common_space2, axis=0))
        else:
            common_space2 /= float(len(X))
        return common_space, common_space2

    return common_space


def compute_template(
    X,
    region,
    kind="searchlight_pca",
    max_npc=None,
    common_topography=True,
    demean=True,
):
    """
    Compute a template from a set of datasets.

    ----------
    Parameters:

       - X (ndarray): The input datasets.
       - region (ndarray or None): The region indices for searchlight or region-based template computation.
       - sl (ndarray or None): The searchlight indices for searchlight-based template computation.
       - region (int or None, optional): The index of the region to consider. If None, all regions are considered (or searchlights). Defaults to None.
       - kind (str): The type of template computation algorithm to use. Can be "pca", "pcav1", "pcav2", or "cls".
       - max_npc (int or None): The maximum number of principal components to use for PCA-based template computation.
       - common_topography (bool): Whether to enforce common topography across datasets.
       - demean (bool): Whether to demean the datasets before template computation.

    ----------
    Returns:
        tmpl : The computed template on all parcels (or searchlights).
    """
    mapping = {
        "searchlight_pca": compute_PCA_template,
        "parcels_pca": compute_PCA_template,
        "cls": compute_procrustes_template,
    }

    if kind == "procrustes":
        tmpl = compute_procrustes_template(
            X=X,
            region=region,
            reflection=True,
            scaling=False,
            zscore_common=True,
        )
    elif kind in mapping:
        tmpl = mapping[kind](X, sl=region, max_npc=max_npc, demean=demean)
    else:
        raise ValueError("Unknown template kind")

    if common_topography:
        if region is not None:
            X_ = X[:, :, region]
        else:
            X_ = np.copy(X)
        ns, nt, nv = X_.shape
        T = procrustes(np.tile(tmpl, (ns, 1)), X_.reshape(ns * nt, nv))
        tmpl = tmpl @ T
    return tmpl.astype(np.float32)
