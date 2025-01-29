import torch
import numpy as np
import pytest
from fmralign.sparse_template_alignment import (
    _rescaled_euclidean_mean_torch,
    _align_images_to_template,
    _fit_sparse_template,
)
from nibabel.nifti1 import Nifti1Image
from fmralign.tests.utils import sample_subjects_data, random_niimg
from itertools import product
from fmralign.alignment_methods import SparseUOT
from fmralign.sparse_template_alignment import SparseTemplateAlignment

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))


@pytest.mark.skip_if_no_mkl
@pytest.mark.parametrize(
    "scale_average, device", product([True, False], devices)
)
def test_rescaled_euclidean_mean_torch(scale_average, device):
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
