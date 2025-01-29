import numpy as np
from nibabel.nifti1 import Nifti1Image
from nilearn.surface import SurfaceImage

from fmralign.sparse_pairwise_alignment import SparsePairwiseAlignment
from fmralign.tests.utils import (
    random_niimg,
    surf_img,
)


def test_surface_alignment():
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


def test_parcellation_retrieval():
    """Test that PairwiseAlignment returns both the\n
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
