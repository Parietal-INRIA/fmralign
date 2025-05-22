from itertools import product

import numpy as np
import pytest
import torch
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers import NiftiMasker

from fmralign.alignment_methods import POTAlignment, SparseUOT
from fmralign.sparse_template_alignment import (
    SparseTemplateAlignment,
    _align_images_to_template,
    _fit_sparse_template,
    _rescaled_euclidean_mean_torch,
)
from fmralign.template_alignment import TemplateAlignment
from fmralign.tests.utils import random_niimg, sample_subjects_data

modalities = ["response", "connectivity", "hybrid"]
devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))


@pytest.mark.skip_if_no_mkl
@pytest.mark.parametrize(
    "scale_average, device", product([True, False], devices)
)
def test_rescaled_euclidean_mean_torch(scale_average, device):
    """Test that _rescaled_euclidean_mean_torch returns a tensor\n
    with the same shape and dtype as the input tensors"""
    subjects_data_np = sample_subjects_data()
    subjects_data = [
        torch.tensor(data, device=device, dtype=torch.float32)
        for data in subjects_data_np
    ]
    average_data = _rescaled_euclidean_mean_torch(subjects_data)
    assert average_data.shape == subjects_data[0].shape
    assert average_data.dtype == subjects_data[0].dtype

    if scale_average is False:
        euclidean_mean = torch.mean(torch.stack(subjects_data), dim=0)
        assert torch.allclose(average_data, euclidean_mean)


@pytest.mark.skip_if_no_mkl
@pytest.mark.parametrize("device", devices)
def test_align_images_to_template(device):
    """Test that _align_images_to_template returns a list of\n
    aligned data and a list of estimators"""
    subjects_data_np = sample_subjects_data()
    subjects_data = [
        torch.tensor(data, device=device, dtype=torch.float32)
        for data in subjects_data_np
    ]
    sparsity_mask = torch.eye(
        subjects_data[0].shape[0], device=device
    ).to_sparse_coo()
    subjects_estimators = [
        SparseUOT(sparsity_mask, device=device)
        for _ in range(len(subjects_data))
    ]
    template = _rescaled_euclidean_mean_torch(subjects_data)
    aligned_data, subjects_estimators = _align_images_to_template(
        subjects_data,
        template,
        subjects_estimators,
    )
    assert len(aligned_data) == len(subjects_data)
    assert len(subjects_estimators) == len(subjects_data)
    assert aligned_data[0].shape == subjects_data[0].shape


@pytest.mark.skip_if_no_mkl
@pytest.mark.parametrize("device", devices)
def test_fit_sparse_template(device):
    """Test that _fit_sparse_template returns a template and\n
    a list of AlignmentEstimator objects"""
    subjects_data_np = sample_subjects_data()
    subjects_data = [
        torch.tensor(data, device=device, dtype=torch.float32)
        for data in subjects_data_np
    ]
    sparsity_mask = torch.eye(
        subjects_data[0].shape[0], device=device
    ).to_sparse_coo()
    template, subjects_estimators = _fit_sparse_template(
        subjects_data,
        sparsity_mask,
        device=device,
    )
    assert template.shape == subjects_data[0].shape
    assert len(subjects_estimators) == len(subjects_data)
    assert template.device == device
    for estimator in subjects_estimators:
        assert estimator.device == device


@pytest.mark.skip_if_no_mkl
def test_parcellation_retrieval():
    """Test that SparseTemplateAlignment returns both the\n
    labels and the parcellation image
    """
    n_pieces = 3
    imgs = [random_niimg((8, 7, 6))[0]] * 3
    alignment = SparseTemplateAlignment(n_pieces=n_pieces)
    alignment.fit(imgs)

    labels, parcellation_image = alignment.get_parcellation()
    assert isinstance(labels, np.ndarray)
    assert len(np.unique(labels)) == n_pieces
    assert isinstance(parcellation_image, Nifti1Image)
    assert parcellation_image.shape == imgs[0].shape[:-1]


@pytest.mark.skip_if_no_mkl
def test_parcellation_before_fit():
    """Test that SparseTemplateAlignment raises an error if\n
    the parcellation is retrieved before fitting
    """
    alignment = SparseTemplateAlignment()
    with pytest.raises(
        AttributeError,
        match="Parcellation has not been computed yet",
    ):
        alignment.get_parcellation()


@pytest.mark.skip_if_no_mkl
def test_consistency_with_dense_templates():
    """Test that SparseTemplateAlignment outputs\n
    consistent templates with TemplateAlignment"""
    img1, mask_img = random_niimg((8, 7, 6, 20))
    img2, _ = random_niimg((8, 7, 6, 20))
    img3, _ = random_niimg((8, 7, 6, 20))
    masker = NiftiMasker(mask_img=mask_img).fit()

    dense_algo = TemplateAlignment(
        n_pieces=3,
        masker=masker,
        alignment_method=POTAlignment(),
    )
    dense_algo.fit([img1, img2, img3])

    # Do not recompute the masker and the clustering
    _, clustering_img = dense_algo.get_parcellation()
    masker = dense_algo.masker
    sparse_algo = SparseTemplateAlignment(
        clustering=clustering_img,
        masker=masker,
    )
    sparse_algo.fit([img1, img2, img3])

    template1 = dense_algo.template
    template2 = sparse_algo.template
    assert np.allclose(
        masker.transform(template1), masker.transform(template2)
    )


@pytest.mark.skip_if_no_mkl
@pytest.mark.parametrize(
    "modality, device",
    product(modalities, devices),
)
def test_various_modalities(modality, device):
    """Test all modalities with Nifti images"""
    n_pieces = 3
    img1, _ = random_niimg((8, 7, 6, 10))
    img2, _ = random_niimg((8, 7, 6, 10))
    img3, _ = random_niimg((8, 7, 6, 10))
    alignment = SparseTemplateAlignment(
        n_pieces=n_pieces, modality=modality, device=device
    )

    # Test fitting
    alignment.fit([img1, img2, img3])
    assert isinstance(alignment.template, Nifti1Image)
    if modality == "response":
        assert alignment.template.shape == img1.shape
    elif modality == "connectivity":
        assert alignment.template.shape[-1] == n_pieces
    elif modality == "hybrid":
        assert alignment.template.shape[-1] == n_pieces + img1.shape[-1]

    # Test transformation on existing subject
    img_transformed = alignment.transform(img1, subject_index=0)
    assert isinstance(img_transformed, Nifti1Image)
    assert img_transformed.shape == img1.shape

    # Test transformation on new subject
    new_img, _ = random_niimg((8, 7, 6, 10))
    img_transformed = alignment.transform(new_img)
    assert isinstance(img_transformed, Nifti1Image)
    assert img_transformed.shape == img1.shape
