from fmralign.alignment_methods import IndividualizedNeuralTuning as BaseINT
from .regions import compute_searchlights, compute_parcels
from nilearn.maskers import NiftiMasker
from nibabel import Nifti1Image
import numpy as np


class IndividualizedNeuralTuning(BaseINT):
    """
    Wrapper for the IndividualTuningModel class to be used in fmralign with Niimg objects.
    Preprocessing and searchlight/parcellation alignment are done without any user input.

    Method of alignment based on the Individualized Neural Tuning model, by Feilong Ma et al. (2023).
    It uses searchlight/parcelation alignment to denoise the data, and then computes the stimulus response matrix.
    See article : https://doi.org/10.1162/imag_a_00032
    """

    def __init__(
        self,
        template="pca",
        decomp_method=None,
        alignment_method="searchlight",
        n_pieces=150,
        searchlight_radius=20,
        n_components=None,
        n_jobs=1,
    ):
        """
        Initialize the IndividualizedNeuralTuning object.

        Parameters:
        -----------

        - tmpl_kind (str): The type of template used for alignment. Default is "pca".
        - decomp_method (str): The decomposition method used for template construction. Default is None.
        - alignment_method (str): The alignment method used. Default is "searchlight".
        - n_pieces (int): The number of pieces to divide the data into if using parcelation. Default is 150.
        - radius (int): The radius of the searchlight sphere in millimeters. Default is 20.
        - latent_dim (int): The number of latent dimensions to use. Default is None.
        - n_jobs (int): The number of parallel jobs to run. Default is 1.
        """
        super().__init__(
            template=template,
            decomp_method=decomp_method,
            n_components=n_components,
            alignment_method=alignment_method,
            n_jobs=n_jobs,
        )
        self.n_pieces = n_pieces
        self.radius = searchlight_radius
        self.mask_img = None
        self.masker = None

    def fit(
        self,
        imgs,
        masker: NiftiMasker = None,
        mask_img: Nifti1Image = None,
        tuning: bool = True,
        y=None,
        verbose=0,
    ):
        """
        Fit the model to the data.
        Can either take as entry a masking image or a masker object.
        This information will be kept for transforming the data.

        Parameters
        ----------
        imgs : list of Nifti1Image
            The images to be fit.
        masker : NiftiMasker
            The masker to be used to transform the images into the common space.
        mask_img : Nifti1Image
            The mask to be used to transform the images into the common space.
        tuning : bool
            Whether to perform tuning or not.
        y : None
            Not used.
        verbose : int
            The verbosity level.
        """

        if mask_img is None:
            mask_img = masker.mask_img
        self.mask_img = mask_img
        self.masker = masker

        X = np.array([masker.transform(img) for img in imgs])

        if self.alignment_method == "searchlight":
            _, searchlights, dists = compute_searchlights(
                niimg=imgs[0],
                mask_img=mask_img,
            )
            super().fit(
                X,
                searchlights,
                dists,
                radius=self.radius,
                tuning=tuning,
                verbose=verbose,
            )

        elif self.alignment_method == "parcellation":
            parcels = compute_parcels(
                niimg=imgs[0],
                mask=masker.mask_img,
                n_parcels=self.n_pieces,
                n_jobs=self.n_jobs,
            )
            super().fit(X, parcels=parcels, tuning=tuning, verbose=verbose)

        return self

    def transform(self, imgs, y=None, verbose=0):
        """
        Transform the data into the common space and return the Nifti1Image objects.

        Parameters
        ----------
        imgs : list of Nifti1Image
            The images to be transformed into the common space.
        y : None
            Not used.
        verbose : int
            The verbosity level.

        Returns
        -------
        preds : list of Nifti1Image
            The images in the common space.
        """
        if self.masker is None:
            self.masker = NiftiMasker(mask_img=self.mask_img)

        X = self.masker.fit_transform(imgs)
        Y = super().transform(X, verbose=verbose)
        preds = []
        for y in Y:
            preds.append(self.masker.inverse_transform(y))

        return preds
