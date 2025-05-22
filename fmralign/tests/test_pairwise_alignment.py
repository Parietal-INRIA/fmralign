# -*- coding: utf-8 -*-
import copy

import numpy as np
import pytest
from nibabel import Nifti1Image
from nilearn.image import new_img_like
from nilearn.maskers import NiftiMasker
from nilearn.surface import SurfaceImage

from fmralign.pairwise_alignment import PairwiseAlignment
from fmralign.tests.utils import (
    assert_algo_transform_almost_exactly,
    random_niimg,
    surf_img,
    zero_mean_coefficient_determination,
)


def test_unsupported_alignment():
    img1, _ = random_niimg((8, 7, 6, 10))
    img2, _ = random_niimg((7, 6, 8, 5))
    args = {"alignment_method": "scaled_procrustes"}
    algo = PairwiseAlignment(**args)
    with pytest.raises(NotImplementedError):
        algo.fit(img1, img2)


def test_pairwise_identity():
    img1, mask_img = random_niimg((8, 7, 6, 10))
    masker = NiftiMasker(mask_img=mask_img).fit()
    args_list = [
        {"alignment_method": "identity", "masker": masker},
        {"alignment_method": "identity", "n_pieces": 3, "masker": masker},
        {
            "alignment_method": "identity",
            "n_pieces": 3,
            "masker": masker,
            "n_jobs": 2,
        },
    ]
    for args in args_list:
        algo = PairwiseAlignment(**args)
        assert_algo_transform_almost_exactly(algo, img1, img1, masker=masker)

    # test intersection of clustering and mask
    data_mask = copy.deepcopy(mask_img.get_fdata())
    data_mask[0] = 0
    # create ground truth
    data_clust = copy.deepcopy(data_mask)
    data_clust[1] = 3
    # create 2-parcels clustering, smaller than background
    clustering = new_img_like(mask_img, data_clust)

    # clustering is smaller than mask
    assert (mask_img.get_fdata() > 0).sum() > (
        clustering.get_fdata() > 0
    ).sum()
    algo = PairwiseAlignment(
        alignment_method="identity", masker=masker, clustering=clustering
    )
    with pytest.warns(UserWarning):
        algo.fit(img1, img1)
    assert (algo.masker.mask_img_.get_fdata() > 0).sum() == (
        clustering.get_fdata() > 0
    ).sum()

    # test warning raised if parcel is 0 :
    null_im = new_img_like(img1, np.zeros_like(img1.get_fdata()))
    with pytest.warns(UserWarning):
        algo.fit(null_im, null_im)


def test_models_against_identity():
    img1, mask_img = random_niimg((7, 6, 8, 10))
    img2, _ = random_niimg((7, 6, 8, 10))
    masker = NiftiMasker(mask_img=mask_img).fit()
    ground_truth = masker.transform(img2)
    identity_baseline_score = zero_mean_coefficient_determination(
        ground_truth, masker.transform(img1)
    )
    for alignment_method in [
        "ridge_cv",
        "scaled_orthogonal",
        "optimal_transport",
        "diagonal",
    ]:
        for clustering in ["kmeans", "hierarchical_kmeans"]:
            algo = PairwiseAlignment(
                alignment_method=alignment_method,
                masker=masker,
                clustering=clustering,
                n_pieces=2,
                n_jobs=1,
            )
            algo.fit(img1, img2)
            im_test = algo.transform(img1)
            algo_score = zero_mean_coefficient_determination(
                ground_truth, masker.transform(im_test)
            )
            assert algo_score >= identity_baseline_score


def test_parcellation_retrieval():
    """Test that PairwiseAlignment returns both the\n
    labels and the parcellation image"""
    n_pieces = 3
    img1, _ = random_niimg((8, 7, 6))
    img2, _ = random_niimg((8, 7, 6))
    alignment = PairwiseAlignment(n_pieces=n_pieces)
    alignment.fit(img1, img2)

    labels, parcellation_image = alignment.get_parcellation()
    assert isinstance(labels, np.ndarray)
    assert len(np.unique(labels)) == n_pieces
    assert isinstance(parcellation_image, Nifti1Image)
    assert parcellation_image.shape == img1.shape


def test_parcellation_before_fit():
    """Test that PairwiseAlignment raises an error if\n
    the parcellation is retrieved before fitting"""
    alignment = PairwiseAlignment()
    with pytest.raises(
        AttributeError, match="Parcellation has not been computed yet"
    ):
        alignment.get_parcellation()


@pytest.mark.parametrize("modality", ["response", "connectivity", "hybrid"])
def test_various_modalities(modality):
    """Test all modalities with Nifti images"""
    n_pieces = 3
    img1, _ = random_niimg((8, 7, 6, 10))
    img2, _ = random_niimg((8, 7, 6, 10))
    alignment = PairwiseAlignment(n_pieces=n_pieces, modality=modality)

    # Test fitting
    alignment.fit(img1, img2)

    # Test transformation
    img_transformed = alignment.transform(img1)
    assert img_transformed.shape == img1.shape
    assert isinstance(img_transformed, Nifti1Image)


@pytest.mark.parametrize("modality", ["response", "connectivity", "hybrid"])
def test_surface_alignment(modality):
    """Test compatibility with `SurfaceImage`"""
    n_pieces = 3
    img1 = surf_img(20)
    img2 = surf_img(20)
    alignment = PairwiseAlignment(n_pieces=n_pieces, modality=modality)

    # Test fitting
    alignment.fit(img1, img2)

    # Test transformation
    img_transformed = alignment.transform(img1)
    assert img_transformed.shape == img1.shape
    assert isinstance(img_transformed, SurfaceImage)

    # Test parcellation retrieval
    labels, parcellation_image = alignment.get_parcellation()
    assert isinstance(labels, np.ndarray)
    assert len(np.unique(labels)) == n_pieces
    assert isinstance(parcellation_image, SurfaceImage)
