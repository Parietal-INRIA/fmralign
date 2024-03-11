"""Piecewise alignment model. This model decomposes the data into regions (pieces).
Those can either be searchlights or parcels (computed with standard parcellation algorithms).
See the ```nilearn``` documentation for more details:
- https://nilearn.github.io/stable/modules/generated/nilearn.regions.Parcellations.html
- https://nilearn.github.io/stable/modules/generated/nilearn.decoding.SearchLight.html
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .regions import (
    template,
    piece_ridge,
    searchlight_weights,
)
from joblib import Parallel, delayed


class PiecewiseAlignment(BaseEstimator, TransformerMixin):
    """Searchlight alignment model. This model decomposes the data into a
    global template and a linear transformation for each subject.
    The global template is computed using a searchlight/parcellation approach.
    The linear transformation is computed using a ridge regression.
    This step is enssential to the hyperalignment model, as it is
    used to remove noise from the raw data.
    """

    def __init__(
        self,
        template_kind="pca",
        common_topography=True,
        verbose=True,
        n_jobs=1,
    ):
        """
        Parameters
        ----------
        alignment_method : str, default="ridge"
            The alignment method to use. Can be "ridge" or "ensemble_ridge".
        template_kind : str, default="pca"
            The kind of template to use. Can be "pca" or "mean".
        demean : bool, default=False
            Whether to demean the data before alignment.
        verbose : bool, default=True
            Whether to display progress bar.
        n_jobs : int, default=-1
        """
        self.W = []
        self.Xhat = []
        self.n_s = None
        self.n_t = None
        self.n_v = None
        self.template_kind = template_kind
        self.verbose = verbose
        self.common_topography = common_topography
        self.n_jobs = n_jobs
        self.regions = None
        self.distances = None
        self.radius = None
        self.weights = None

    def compute_linear_transformation(self, data, template):
        """Compute the linear transformation for a given subject provided the global template.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_voxels)
            The brain images for one subject.
            Those are the B_1, ..., B_n in the paper.
        template : ndarray of shape (n_samples, n_voxels)
            The global template M.

        Returns
        -------
        Xhat : ndarray of shape (n_samples, n_voxels)
            The denoised estimation signal for each subject.
        """

        x_hat = piece_ridge(
            X=template,
            Y=data,
            alpha=10,
            regions=self.regions,
            verbose=self.verbose,
        )
        return x_hat

    def fit_transform(
        self,
        X: np.ndarray,
        regions=None,
        dists=None,
        radius=None,
        weights=None,
    ):
        """From given fmri data, compute the global template and the linear transformation.
        This provides denoised signal estimations using template alignment.

        Parameters
        ----------
        X : list of ndarray of shape (n_samples, n_voxels)
            The brain images for one subject.
        searchlights : list of searchlights
            The searchlight indices.
        dists : list of distances

        radius : int
            The radius of the searchlight (in millimeters)

        Returns
        -------
        Xhat : list of ndarray of shape (n_samples, n_voxels)
            The denoised estimations B_1, ... B_p for each subject.
        """

        self.FUNC = "RegionAlignment"
        if dists is not None and radius is not None:
            self.FUNC = "SearchlightAlignment"

        if self.verbose:
            print(f"[{self.FUNC}] Shape of input data: ", X.shape)

        self.n_s, self.n_t, self.n_v = X.shape
        self.regions = regions
        self.FUNC = "ParcelAlignment"

        if weights is None and dists is not None and radius is not None:
            self.distances = dists
            self.radius = radius
            self.FUNC = "SearchlightAlignment"

        # Compute global template M (sl_template)
        if self.verbose:
            print(f"[{self.FUNC}]Computing global template M ...")

        if dists is None or radius is None:
            self.weights = weights
        elif weights is None:
            self.weights = searchlight_weights(
                searchlights=regions, dists=dists, radius=radius
            )
        else:
            self.weights = weights

        sl_template = template(
            X,
            regions=regions,
            n_jobs=self.n_jobs,
            template_kind=self.template_kind,
            common_topography=self.common_topography,
            weights=self.weights,
        )

        self.Xhat = Parallel(n_jobs=self.n_jobs)(
            delayed(self.compute_linear_transformation)(X[i], sl_template)
            for i in range(self.n_s)
        )
        return np.array(self.Xhat)
