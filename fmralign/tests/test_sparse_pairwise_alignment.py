import numpy as np
import pytest
import torch
from nibabel.nifti1 import Nifti1Image
from nilearn.surface import SurfaceImage

from fmralign.sparse_pairwise_alignment import SparsePairwiseAlignment
from fmralign.tests.utils import (
    random_niimg,
    surf_img,
)

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))


@pytest.mark.skip_if_no_mkl
@pytest.mark.parametrize("device", devices)
def test_fit_method(device):
    """Test various solvers for SparsePairwiseAlignment"""
    alignment = SparsePairwiseAlignment(n_pieces=3, device=device)
    img1, _ = random_niimg((8, 7, 6, 10))
    img2, _ = random_niimg((8, 7, 6, 10))
    alignment.fit(img1, img2)
    assert alignment.device == device


@pytest.mark.skip_if_no_mkl
@pytest.mark.parametrize("device", devices)
def test_identity_alignment(device):
    """Test the identity alignment for SparsePairwiseAlignment"""
    alignment = SparsePairwiseAlignment(device=device, reg=1e-6, tol=1e-10)
    img, _ = random_niimg((8, 7, 6, 100))
    alignment.fit(img, img)
    img_transformed = alignment.transform(img)
    assert img_transformed.shape == img.shape
    assert isinstance(img_transformed, Nifti1Image)
    masker = alignment.masker
    assert np.allclose(
        masker.transform(img), masker.transform(img_transformed)
    )


@pytest.mark.skip_if_no_mkl
@pytest.mark.parametrize(
    "device",
    devices,
)
def test_surface_alignment(device):
    """Test compatibility with `SurfaceImage`"""
    n_pieces = 3
    img1 = surf_img(20)
    img2 = surf_img(20)
    alignment = SparsePairwiseAlignment(n_pieces=n_pieces)

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


@pytest.mark.skip_if_no_mkl
def test_parcellation_retrieval():
    """Test that SparsePairwiseAlignment returns both the\n
    labels and the parcellation image"""
    n_pieces = 3
    img1, _ = random_niimg((8, 7, 6))
    img2, _ = random_niimg((8, 7, 6))
    alignment = SparsePairwiseAlignment(n_pieces=n_pieces)
    alignment.fit(img1, img2)

    labels, parcellation_image = alignment.get_parcellation()
    assert isinstance(labels, np.ndarray)
    assert len(np.unique(labels)) == n_pieces
    assert isinstance(parcellation_image, Nifti1Image)
    assert parcellation_image.shape == img1.shape


@pytest.mark.skip_if_no_mkl
def test_parcellation_before_fit():
    """Test that SparsePairwiseAlignment raises an error if\n
    the parcellation is retrieved before fitting"""
    alignment = SparsePairwiseAlignment()
    with pytest.raises(
        AttributeError, match="Parcellation has not been computed yet"
    ):
        alignment.get_parcellation()
