"""Module for sparse pairwise functional alignment."""

import torch
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from fmralign._utils import _create_sparse_cluster_matrix
from fmralign.alignment_methods import SparseUOT
from fmralign.preprocessing import ParcellationMasker


class SparsePairwiseAlignment(BaseEstimator, TransformerMixin):
    """Decompose the source and target images into regions and align
    corresponding regions independently."""

    def __init__(
        self,
        alignment_method="sparse_uot",
        n_pieces=1,
        clustering="kmeans",
        mask=None,
        smoothing_fwhm=None,
        standardize=False,
        detrend=False,
        target_affine=None,
        target_shape=None,
        low_pass=None,
        high_pass=None,
        t_r=None,
        memory=Memory(location=None),
        memory_level=0,
        device="cpu",
        n_jobs=1,
        verbose=0,
        **kwargs,
    ):
        """If n_pieces > 1, decomposes the images into regions and align each
        source/target region independantly.

        Parameters
        ----------
        alignment_method: string
            Algorithm used to perform alignment between X_i and Y_i :
            * either 'identity', 'scaled_orthogonal', 'optimal_transport',
            'ridge_cv', 'permutation', 'diagonal'
            * or an instance of one of alignment classes
            (imported from fmralign.alignment_methods)
        n_pieces: int, optional (default = 1)
            Number of regions in which the data is parcellated for alignment.
            If 1 the alignment is done on full scale data.
            If >1, the voxels are clustered and alignment is performed
            on each cluster applied to X and Y.
        clustering : string or 3D Niimg optional (default : kmeans)
            'kmeans', 'ward', 'rena', 'hierarchical_kmeans' method used for
            clustering of voxels based on functional signal, passed to
            nilearn.regions.parcellations
            If 3D Niimg, image used as predefined clustering,
            n_pieces is then ignored.
        mask: Niimg-like object, instance of NiftiMasker or
                                MultiNiftiMasker, optional (default = None)
            Mask to be used on data. If an instance of masker is passed,
            then its mask will be used. If no mask is given,
            it will be computed automatically by a MultiNiftiMasker
            with default parameters.
        smoothing_fwhm: float, optional (default = None)
            If smoothing_fwhm is not None, it gives the size in millimeters
            of the spatial smoothing to apply to the signal.
        standardize: boolean, optional (default = False)
            If standardize is True, the time-series are centered and normed:
            their variance is put to 1 in the time dimension.
        detrend: boolean, optional (default = None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details
        target_affine: 3x3 or 4x4 matrix, optional (default = None)
            This parameter is passed to nilearn.image.resample_img.
            Please see the related documentation for details.
        target_shape: 3-tuple of integers, optional (default = None)
            This parameter is passed to nilearn.image.resample_img.
            Please see the related documentation for details.
        low_pass: None or float, optional (default = None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details.
        high_pass: None or float, optional (default = None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details.
        t_r: float, optional (default = None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details.
        memory: instance of joblib.Memory or string (default = None)
            Used to cache the masking process and results of algorithms.
            By default, no caching is done. If a string is given, it is the
            path to the caching directory.
        memory_level: integer, optional (default = None)
            Rough estimator of the amount of memory used by caching.
            Higher value means more memory for caching.
        n_jobs: integer, optional (default = 1)
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        verbose: integer, optional (default = 0)
            Indicate the level of verbosity. By default, nothing is printed.
        """
        self.n_pieces = n_pieces
        self.alignment_method = alignment_method
        self.clustering = clustering
        self.mask = mask
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.device = device
        self.verbose = verbose
        self.kwargs = kwargs

    def fit(self, X, Y):
        """Fit data X and Y and learn transformation to map X to Y.

        Parameters
        ----------
        X: Niimg-like object
            Source data.

        Y: Niimg-like object
            Target data

        Returns
        -------
        self
        """
        self.parcel_masker = ParcellationMasker(
            n_pieces=self.n_pieces,
            clustering=self.clustering,
            mask=self.mask,
            smoothing_fwhm=self.smoothing_fwhm,
            standardize=self.standardize,
            detrend=self.detrend,
            low_pass=self.low_pass,
            high_pass=self.high_pass,
            t_r=self.t_r,
            target_affine=self.target_affine,
            target_shape=self.target_shape,
            memory=self.memory,
            memory_level=self.memory_level,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        self.parcel_masker.fit([X, Y])
        self.masker = self.parcel_masker.masker_
        self.mask = self.parcel_masker.masker_.mask_img_
        self.labels_ = self.parcel_masker.labels
        self.n_pieces = self.parcel_masker.n_pieces

        X = torch.tensor(self.masker.transform(X), device=self.device)
        Y = torch.tensor(self.masker.transform(Y), device=self.device)

        sparsity_mask = _create_sparse_cluster_matrix(self.labels_)
        if self.alignment_method == "sparse_uot":
            self.fit_ = SparseUOT(
                sparsity_mask=sparsity_mask,
                device=self.device,
                verbose=self.verbose,
                **self.kwargs,
            ).fit(X, Y)
        else:
            raise ValueError(
                f"Unknown alignment method: {self.alignment_method}"
            )

        return self

    def transform(self, img):
        """Predict data from X.

        Parameters
        ----------
        img: Niimg-like object
            Source data

        Returns
        -------
        transformed_img: Niimg-like object
            Predicted data
        """
        if not hasattr(self, "fit_"):
            raise ValueError(
                "This instance has not been fitted yet. "
                "Please call 'fit' before 'transform'."
            )
        X = torch.tensor(self.masker.transform(img), device=self.device)
        transformed_data = self.fit_.transform(X)
        transformed_img = self.masker.inverse_transform(
            transformed_data.cpu().detach().numpy()
        )
        return transformed_img

    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here.

        Will raise AttributeError if called.
        """
        raise AttributeError(
            "type object 'PairwiseAlignment' has no attribute 'fit_transform'"
        )

    def get_parcellation(self):
        """Get the parcellation masker used for alignment.

        Returns
        -------
        labels: `list` of `int`
            Labels of the parcellation masker.
        parcellation_img: Niimg-like object
            Parcellation image.
        """
        if hasattr(self, "parcel_masker"):
            check_is_fitted(self)
            labels = self.parcel_masker.get_labels()
            parcellation_img = self.parcel_masker.get_parcellation_img()
            return labels, parcellation_img
        else:
            raise AttributeError(
                (
                    "Parcellation has not been computed yet,"
                    "please fit the alignment estimator first."
                )
            )
