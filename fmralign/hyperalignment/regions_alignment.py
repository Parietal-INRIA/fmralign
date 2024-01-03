import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .regions import (
    template,
    piece_ridge,
    searchlight_weights,
)
import os
from joblib import Parallel, delayed


class RegionAlignment(BaseEstimator, TransformerMixin):
    """Searchlight alignment model. This model decomposes the data into a
    global template and a linear transformation for each subject.
    The global template is computed using a searchlight approach.
    The linear transformation is computed using a ridge regression.
    This step is enssential to the hyperalignment model, as it is
    used as a denoiser for the data.
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

    def __init__(
        self,
        alignment_method="searchlight_ridge",
        template_kind="searchlight_pca",
        verbose=True,
        path="cache/int/",
        cache=True,
        n_jobs=-1,
    ):
        self.W = []
        self.Xhat = []
        self.n_s = None
        self.n_t = None
        self.n_v = None
        self.warp_alignment_method = alignment_method
        self.template_kind = template_kind
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.regions = None
        self.distances = None
        self.radius = None
        self.weights = None
        self.path = path
        self.cache = (path is not None) and (cache)

        if self.cache:
            # Check if cache folder exists
            if not os.path.exists(self.path):
                os.makedirs(self.path)

    def compute_linear_transformation(
        self, x_i, template, i: int = 0, save=True
    ):
        """Compute the linear transformation W_i for a given subject.
        ----------
        Parameters
        ----------
        x_i : ndarray of shape (n_samples, n_voxels)
            The brain images for one subject.
            Those are the B_1, ..., B_n in the paper.
        template : ndarray of shape (n_samples, n_voxels)
            The global template M.

        Returns
        -------
        Xhat : ndarray of shape (n_samples, n_voxels)
            The denoised estimation signal for each subject.
        """
        try:
            W_p = np.load(self.path + (f"/train_data_W_{i}.npy"))
            if self.verbose:
                print(f"Loaded W_{i} from cache")
            x_hat = template.dot(W_p)
            del W_p
            return x_hat

        except:  # noqa E722
            if self.verbose:
                print(f"No cache found, computing W_{i}")

            x_hat = piece_ridge(
                X=x_i,
                Y=template,
                regions=self.regions,
                weights=self.weights,
                verbose=self.verbose,
            )

            if self.cache:
                np.save(self.path + (f"/train_data_W_{i}.npy"), x_hat)
        return x_hat

    def fit_transform(
        self,
        X: np.ndarray,
        regions=None,
        dists=None,
        radius=None,
        weights=None,
        id=None,
    ):
        """From brain imgs compute the INT model (M, Ws, S)
          with the given parameters)

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

        try:
            self.Xhat = np.load(self.path + "/train_data_denoised.npy")
            if self.verbose:
                print(f"[{self.FUNC}] Loaded denoised data from cache")
            return self.Xhat
        except:  # noqa E722
            if self.verbose:
                print(f"[{self.FUNC}] No cache found, computing denoised data")

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

        try:
            sl_template = np.load(self.path + ("/train_data_template.npy"))
            if self.verbose:
                print("Loaded template from cache")
        except:  # noqa E722
            if self.verbose:
                print(f"[{self.FUNC}] No cache found, computing template")
            if dists is None or radius is None:
                self.weights = None
            else:
                self.weights = searchlight_weights(
                    searchlights=regions, dists=dists, radius=radius
                )
            sl_template = template(
                X,
                regions=regions,
                n_jobs=self.n_jobs,
                template_kind=self.template_kind,
                verbose=self.verbose,
                weights=self.weights,
            )

        if self.cache:
            np.save(self.path + ("/train_data_template.npy"), sl_template)
            if self.verbose:
                print(f"[{self.FUNC}] Saved template to cache")

        self.Xhat = Parallel(n_jobs=self.n_jobs)(
            delayed(self.compute_linear_transformation)(X[i], sl_template, i)
            for i in range(self.n_s)
        )

        if id is None:
            id = np.random.randint(0, 1000000)

        if self.cache:
            np.save(self.path + ("/train_data_denoised.npy"), self.Xhat)

        return np.array(self.Xhat)

    def get_linear_transformations(self):
        """Return the linear transformations W_1, ... W_p for each subject.
        ----------
        Returns
        -------
        W : list of ndarray of shape (n_voxels, n_voxels)
            The linear transformations W_1, ... W_p for each subject.
        """
        return np.array(self.W)

    def get_denoised_estimation(self):
        """Return the denoised estimations B_1, ... B_p for each subject."""
        return self.Xhat
