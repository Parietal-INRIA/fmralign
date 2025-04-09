from itertools import product

import numpy as np
import pytest
import torch
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers import NiftiMasker
from numpy.testing import assert_array_almost_equal

from fmralign.alignment_methods import POTAlignment, SparseOT
from fmralign.sparse_template_alignment import (
    SparseTemplateAlignment,
    _align_images_to_template,
    _fit_sparse_template,
    _rescaled_euclidean_mean_torch,
)
from fmralign.template_alignment import TemplateAlignment
from fmralign.tests.utils import random_niimg, sample_subjects_data

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))


@pytest.mark.parametrize(
    "scale_average, device", product([True, False], devices)
)
def test_rescaled_euclidean_mean_torch(scale_average, device):
    """Test that _rescaled_euclidean_mean_torch returns a tensor\n
    with the same shape and dtype as the input tensors"""
    subjects_data_np = sample_subjects_data()
    subjects_data = [
        torch.tensor(data, device=device, dtype=torch.float64)
        for data in subjects_data_np
    ]
    average_data = _rescaled_euclidean_mean_torch(subjects_data)
    assert average_data.shape == subjects_data[0].shape
    assert average_data.dtype == subjects_data[0].dtype

    if scale_average is False:
        euclidean_mean = torch.mean(torch.stack(subjects_data), dim=0)
        assert torch.allclose(average_data, euclidean_mean)


@pytest.mark.parametrize("device", devices)
def test_align_images_to_template(device):
    """Test that _align_images_to_template returns a list of\n
    aligned data and a list of estimators"""
    subjects_data_np = sample_subjects_data()
    subjects_data = [
        torch.tensor(data, device=device, dtype=torch.float64)
        for data in subjects_data_np
    ]
    sparsity_mask = torch.eye(
        subjects_data[0].shape[0], device=device
    ).to_sparse_coo()
    subjects_estimators = [
        SparseOT(sparsity_mask, device=device)
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


@pytest.mark.parametrize("device", devices)
def test_fit_sparse_template(device):
    """Test that _fit_sparse_template returns a template and\n
    a list of AlignmentEstimator objects"""
    subjects_data_np = sample_subjects_data()
    subjects_data = [
        torch.tensor(data, device=device, dtype=torch.float64)
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


def test_consistency_with_dense_templates():
    """Test that SparseTemplateAlignment outputs\n
    consistent templates with TemplateAlignment"""
    img1, mask_img = random_niimg((4, 3, 2, 20))
    img2, _ = random_niimg((4, 3, 2, 20))
    img3, _ = random_niimg((4, 3, 2, 20))
    masker = NiftiMasker(mask_img=mask_img).fit()

    dense_algo = TemplateAlignment(
        n_pieces=3,
        masker=masker,
        alignment_method=POTAlignment(),
    )
    dense_algo.fit([img1, img2, img3])

    sparse_algo = SparseTemplateAlignment(
        n_pieces=3,
        masker=masker,
    )
    sparse_algo.fit([img1, img2, img3])

    # Check that the templates are the same
    template1 = dense_algo.template_img
    template2 = sparse_algo.template_img
    assert_array_almost_equal(
        masker.transform(template1),
        masker.transform(template2),
    )

    # Check that the transformed images are the same
    for i, img in enumerate([img1, img2, img3]):
        img_dense_transformed = dense_algo.transform(img, subject_index=i)
        img_sparse_transformed = sparse_algo.transform(img, subject_index=i)
        assert_array_almost_equal(
            masker.transform(img_dense_transformed),
            masker.transform(img_sparse_transformed),
        )
