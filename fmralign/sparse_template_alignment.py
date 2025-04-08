"""Module for sparse template alignment."""

import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from fmralign._utils import (
    _sparse_cluster_matrix,
)
from fmralign.alignment_methods import SparseOT
from fmralign.preprocessing import ParcellationMasker
from fmralign.sparse_pairwise_alignment import SparsePairwiseAlignment


def _rescaled_euclidean_mean_torch(subjects_data, scale_average=False):
    """Compute the rescaled euclidean mean of the subjects data.

    Parameters
    ----------
    subjects_data : list of torch.Tensor of shape (n_samples, n_features)
        List of subjects data.
    scale_average : bool, optional
        Scale the euclidean template, by default False

    Returns
    -------
    average_data : torch.Tensor of shape (n_samples, n_features)
        Rescaled euclidean mean of the subjects data.
    """
    average_data = torch.mean(torch.stack(subjects_data), dim=0)
    scale = 1
    if scale_average:
        X_norm = 0
        for data in subjects_data:
            X_norm += torch.linalg.norm(data)
        X_norm /= len(subjects_data)
        scale = X_norm / torch.linalg.norm(average_data)
    average_data *= scale

    return average_data


def _align_images_to_template(
    subjects_data,
    template,
    subjects_estimators,
):
    """Align the subjects data to the template using sparse alignment.

    Parameters
    ----------
    subjects_data : List of torch.Tensor of shape (n_samples, n_features)
        List of subjects data.
    template : torch.Tensor of shape (n_samples, n_features)
        Template data.
    subjects_estimators : List of alignment_methods.Alignment
        List of sparse alignment estimators.

    Returns
    -------
    Tuple of List of torch.Tensor of shape (n_samples, n_features)
        and List of alignment_methods.Alignment
        Updated subjects data and alignment estimators.
    """
    n_subjects = len(subjects_data)
    aligned_data = [None] * n_subjects
    for i in range(n_subjects):
        sparse_estimator = subjects_estimators[i]
        sparse_estimator.fit(subjects_data[i], template)
        aligned_data[i] = sparse_estimator.transform(subjects_data[i])
        # Update the estimator in the list
        subjects_estimators[i] = sparse_estimator
    return aligned_data, subjects_estimators


def _fit_sparse_template(
    subjects_data,
    sparsity_mask,
    alignment_method="sparse_uot",
    n_iter=2,
    scale_template=False,
    verbose=False,
    **kwargs,
):
    """Fit a the template to the subjects data using sparse alignment.

    Parameters
    ----------
    subjects_data : list of torch.Tensor of shape (n_samples, n_features)
        List of subjects data.
    sparsity_mask : torch sparse COO tensor
        Sparsity mask for the alignment matrix.
    alignment_method : str, optional
        Sparse alignment method, by default "sparse_uot"
    n_iter : int, optional
        Number of template updates, by default 2
    scale_template : bool, optional
        Scale the template data at each iteration, by default False
    verbose : bool, optional
        Verbosity level, by default False

    Returns
    -------
    Tuple[torch.Tensor, List[alignment_methods.Alignment]]
        Template data and list of alignment estimators
        from the subjects data to the template.

    Raises
    ------
    ValueError
        Unknown alignment method.
    """
    n_subjects = len(subjects_data)
    aligned_data = subjects_data
    if alignment_method != "sparse_uot":
        raise ValueError(f"Unknown alignment method: {alignment_method}")
    subjects_estimators = [
        SparseOT(sparsity_mask, verbose=verbose, **kwargs)
        for _ in range(n_subjects)
    ]
    for _ in range(n_iter):
        template = _rescaled_euclidean_mean_torch(aligned_data, scale_template)
        aligned_data, subjects_estimators = _align_images_to_template(
            subjects_data,
            template,
            subjects_estimators,
        )

    return template, subjects_estimators


class SparseTemplateAlignment(BaseEstimator, TransformerMixin):
    """
    Decompose the source images into regions and summarize subjects information
    in a template, then use pairwise alignment to predict
    new contrast for target subject.
    """

    def __init__(
        self,
        alignment_method="sparse_uot",
        n_pieces=1,
        clustering="kmeans",
        scale_template=False,
        n_iter=2,
        save_template=None,
        masker=None,
        device="cpu",
        n_jobs=1,
        verbose=0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        alignment_method: string
            Algorithm used to perform alignment between source images
            and template, currently only 'sparse_uot' is supported.
        n_pieces: int, optional (default = 1)
            Number of regions in which the data is parcellated for alignment.
            If 1 the alignment is done on full scale data.
            If > 1, the voxels are clustered and alignment is performed on each
            cluster applied to X and Y.
        clustering : string or 3D Niimg optional (default : kmeans)
            'kmeans', 'ward', 'rena', 'hierarchical_kmeans' method used for
            clustering of voxels based on functional signal,
            passed to nilearn.regions.parcellations
            If 3D Niimg, image used as predefined clustering,
            n_pieces is then ignored.
        scale_template: boolean, default False
            rescale template after each inference so that it keeps
            the same norm as the average of training images.
        n_iter: int
           number of iteration in the alternate minimization. Each img is
           aligned n_iter times to the evolving template. If n_iter = 0,
           the template is simply the mean of the input images.
        save_template: None or string(optional)
            If not None, path to which the template will be saved.
        device: string, optional (default = 'cpu')
            Device on which the computation will be done. If 'cuda', the
            computation will be done on the GPU if available.
        n_jobs: integer, optional (default = 1)
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        verbose: integer, optional (default = 0)
            Indicate the level of verbosity. By default, nothing is printed.

        """
        self.template_img = None
        self.alignment_method = alignment_method
        self.n_pieces = n_pieces
        self.clustering = clustering
        self.n_iter = n_iter
        self.scale_template = scale_template
        self.save_template = save_template
        self.masker = masker
        self.device = device
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.kwargs = kwargs

    def fit(self, imgs):
        """
        Learn a template from source images, using alignment.

        Parameters
        ----------
        imgs: List of 4D Niimg-like or List of lists of 3D Niimg-like
            Source subjects data. Each element of the parent list is one subject
            data, and all must have the same length (n_samples).

        Returns
        -------
        self

        Attributes
        ----------
        self.template_img: 4D Niimg object
            Length : n_samples

        """

        self.parcel_masker = ParcellationMasker(
            n_pieces=self.n_pieces,
            clustering=self.clustering,
            masker=self.masker,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        self.parcel_masker.fit(imgs)
        self.masker = self.parcel_masker.masker
        self.labels_ = self.parcel_masker.labels
        self.n_pieces = self.parcel_masker.n_pieces

        subjects_data = [
            torch.tensor(
                self.masker.transform(img),
                device=self.device,
                dtype=torch.float64,
            )
            for img in imgs
        ]
        sparsity_mask = _sparse_cluster_matrix(self.labels_).to(self.device)
        template_data, self.fit_ = _fit_sparse_template(
            subjects_data=subjects_data,
            sparsity_mask=sparsity_mask,
            n_iter=self.n_iter,
            scale_template=self.scale_template,
            alignment_method=self.alignment_method,
            device=self.device,
            verbose=True if self.verbose > 0 else False,
            **self.kwargs,
        )

        self.template_img = self.masker.inverse_transform(
            template_data.cpu().numpy()
        )
        if self.save_template is not None:
            self.template_img.to_filename(self.save_template)

    def transform(self, img, subject_index=None):
        """
        Transform a (new) subject image into the template space.

        Parameters
        ----------
        img: 4D Niimg-like object
            Subject image.
        subject_index: int, optional (default = None)
            Index of the subject to be transformed. It should
            correspond to the index of the subject in the list of
            subjects used to fit the template. If None, a new
            `PairwiseAlignment` object is fitted between the new
            subject and the template before transforming.


        Returns
        -------
        predicted_imgs: 4D Niimg object
            Transformed data.

        """
        if not hasattr(self, "fit_"):
            raise ValueError(
                "This instance has not been fitted yet. "
                "Please call 'fit' before 'transform'."
            )

        if subject_index is None:
            alignment_estimator = SparsePairwiseAlignment(
                n_pieces=self.n_pieces,
                alignment_method=self.alignment_method,
                clustering=self.parcel_masker.get_parcellation_img(),
                masker=self.masker,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
            alignment_estimator.fit(img, self.template_img)
            return alignment_estimator.transform(img)
        else:
            X = torch.tensor(
                self.masker.transform(img),
                device=self.device,
                dtype=torch.float64,
            )
            sparse_estimator = self.fit_[subject_index]
            X_transformed = sparse_estimator.transform(X).cpu().numpy()
            return self.masker.inverse_transform(X_transformed)

    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here. Will raise AttributeError if called."""
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
