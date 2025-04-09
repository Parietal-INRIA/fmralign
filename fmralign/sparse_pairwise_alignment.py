"""Module for sparse pairwise functional alignment."""

import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from fmralign._utils import _sparse_cluster_matrix
from fmralign.alignment_methods import SparseOT
from fmralign.preprocessing import ParcellationMasker


class SparsePairwiseAlignment(BaseEstimator, TransformerMixin):
    """Decompose the source and target images into regions and align
    corresponding regions independently."""

    def __init__(
        self,
        alignment_method="sparse_uot",
        n_pieces=1,
        clustering="kmeans",
        masker=None,
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
            Algorithm used to perform alignment between X and Y.
            Currently, only 'sparse_uot' is available.
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
        masker : None or :class:`~nilearn.maskers.NiftiMasker` or \
                :class:`~nilearn.maskers.MultiNiftiMasker`, or \
                :class:`~nilearn.maskers.SurfaceMasker` , optional
            A mask to be used on the data. If provided, the mask
            will be used to extract the data. If None, a mask will
            be computed automatically with default parameters.
        device: string, optional (default = 'cpu')
            Device on which the computation will be done. If 'cuda', the
            computation will be done on the GPU if available.
        n_jobs: integer, optional (default = 1)
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        verbose: integer, optional (default = 0)
            Indicate the level of verbosity. By default, nothing is printed.
        """
        self.n_pieces = n_pieces
        self.alignment_method = alignment_method
        self.clustering = clustering
        self.masker = masker
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
            masker=self.masker,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        self.parcel_masker.fit([X, Y])
        self.masker = self.parcel_masker.masker
        self.labels_ = self.parcel_masker.labels
        self.n_pieces = self.parcel_masker.n_pieces

        X = torch.tensor(
            self.masker.transform(X), device=self.device, dtype=torch.float64
        )
        Y = torch.tensor(
            self.masker.transform(Y), device=self.device, dtype=torch.float64
        )

        sparsity_mask = _sparse_cluster_matrix(self.labels_)
        if self.alignment_method == "sparse_uot":
            self.fit_ = SparseOT(
                sparsity_mask=sparsity_mask,
                device=self.device,
                verbose=True if self.verbose > 0 else False,
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
        X = torch.tensor(
            self.masker.transform(img), device=self.device, dtype=torch.float64
        )
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
