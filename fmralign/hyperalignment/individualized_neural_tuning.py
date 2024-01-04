"""
A wrapper for the IndividualTuningModel class to be used in fmralign (taking Nifti1 images as input).
"""
from .model import INTEstimator
from .regions import compute_searchlights
from nilearn.maskers import NiftiMasker
from nibabel import Nifti1Image
import numpy as np


class IndividualizedNeuralTuning(INTEstimator):
    def __init__(self, method="searchlight", n_jobs=-1):
        """
        A wrapper for the IndividualTuningModel class to be used in fmralign (taking Nifti1 images as input).
        Method of alignment based on the Individualized Neural Tuning model, by Feilong Ma et al. (2023).
        It uses searchlight/parcelation alignment to denoise the data, and then computes the stimulus response matrix.
        See article : https://doi.org/10.1162/imag_a_00032

        Parameters:
        --------
        - method (str): The method used for hyperalignment. Can be either "searchlight" or "parcellation". Default is "searchlight".
        - n_jobs (int): The number of parallel jobs to run. Default is -1, which uses all available processors.

        Returns:
        --------
        None
        """
        super().__init__(n_jobs=n_jobs)
        self.mask_img = None
        self.masker = None
        self.method = method

    def fit(
        self,
        imgs,
        masker: NiftiMasker = None,
        mask_img: Nifti1Image = None,
        y=None,
        verbose=0,
    ):
        """
        Fit the model to the data.
        Can either take as entry a masking image or a masker object.
        This information will be kept for transforming the data.

        Parameters
        ----------
        imgs : list of Nifti1Image or np.ndarray
            The images to be fit.
        masker : NiftiMasker
            The masker to be used to transform the images into the common space.
        mask_img : Nifti1Image
            The mask to be used to transform the images into the common space.
        y : None
            Not used.
        verbose : int
            The verbosity level.
        """

        if mask_img is None:
            mask_img = masker.mask_img
        self.mask_img = mask_img
        self.masker = masker

        if self.method == "searchlight":
            data, searchlights, dists = compute_searchlights(
                niimg=imgs[0],
                mask_img=mask_img,
            )

        elif self.method == "parcellation":
            raise NotImplementedError("Parcellation method not implemented yet.")

        if isinstance(imgs, np.ndarray):
            X = imgs
        else:
            X = data

        super().fit(X, searchlights, dists, verbose=verbose)
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
