import numpy as np
import pytest
from joblib import Memory
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers import NiftiMasker

from fmralign._utils import ParceledData
from fmralign.preprocessing import ParcellationMasker
from fmralign.tests.utils import random_niimg, surf_img


def test_init_default_params():
    """Test that ParcellationMasker initializes with default parameters"""
    parcel_masker = ParcellationMasker()
    assert parcel_masker.n_pieces == 1
    assert parcel_masker.clustering == "kmeans"
    assert parcel_masker.mask is None
    assert parcel_masker.smoothing_fwhm is None
    assert parcel_masker.standardize == "zscore_sample"
    assert parcel_masker.detrend is False
    assert parcel_masker.labels is None


def test_init_custom_params():
    """Test that ParcellationMasker initializes with custom parameters"""
    parcel_masker = ParcellationMasker(
        n_pieces=2, clustering="ward", standardize=True, detrend=True, n_jobs=2
    )
    assert parcel_masker.n_pieces == 2
    assert parcel_masker.clustering == "ward"
    assert parcel_masker.standardize is True
    assert parcel_masker.detrend is True
    assert parcel_masker.n_jobs == 2


def test_fit_single_image():
    """Test that ParcellationMasker fits a single image"""
    img, _ = random_niimg((8, 7, 6))
    parcel_masker = ParcellationMasker(n_pieces=2)
    parcel_masker.fit(img)

    assert hasattr(parcel_masker, "masker")
    assert parcel_masker.labels is not None
    assert isinstance(parcel_masker.labels, np.ndarray)
    assert len(np.unique(parcel_masker.labels)) == 2  # n_pieces=2


def test_fit_multiple_images():
    """Test that ParcellationMasker fits multiple images"""
    imgs = [random_niimg((8, 7, 6))[0]] * 3
    parcel_masker = ParcellationMasker(n_pieces=2)
    parcel_masker = parcel_masker.fit(imgs)

    assert hasattr(parcel_masker, "masker")
    assert parcel_masker.labels is not None


def test_transform_single_image():
    """Test that ParcellationMasker transforms a single image"""
    img, _ = random_niimg((8, 7, 6))
    parcel_masker = ParcellationMasker(n_pieces=2)
    parcel_masker.fit(img)
    transformed_data = parcel_masker.transform(img)

    assert isinstance(transformed_data, list)
    assert len(transformed_data) == 1
    assert isinstance(transformed_data[0], ParceledData)


def test_transform_multiple_images():
    """Test that ParcellationMasker transforms multiple images"""
    imgs = [random_niimg((8, 7, 6))[0]] * 3
    parcel_masker = ParcellationMasker(n_pieces=2)
    parcel_masker.fit(imgs)
    transformed_data = parcel_masker.transform(imgs)

    assert isinstance(transformed_data, list)
    assert len(transformed_data) == 3
    assert all(
        isinstance(parceled_data, ParceledData)
        for parceled_data in transformed_data
    )


def test_get_labels_before_fit():
    """Test that ParcellationMasker raises ValueError if get_labels is called before fit"""
    parcel_masker = ParcellationMasker()
    with pytest.raises(ValueError, match="Labels have not been computed yet"):
        parcel_masker.get_labels()


def test_get_labels_after_fit():
    img, _ = random_niimg((8, 7, 6))
    parcel_masker = ParcellationMasker(n_pieces=2)
    parcel_masker.fit(img)
    labels = parcel_masker.get_labels()

    assert labels is not None
    assert isinstance(labels, np.ndarray)
    assert len(np.unique(labels)) == 2


def test_different_shaped_images():
    """Test that ParcellationMasker raises NotImplementedError for \
        images of different shapes"""
    img, _ = random_niimg((8, 7, 6))
    # Create image with different shape
    different_data = np.random.rand(8, 7, 8)
    different_img = Nifti1Image(different_data, np.eye(4))

    imgs = [img, different_img]
    parcel_masker = ParcellationMasker()

    with pytest.raises(
        NotImplementedError,
        match="fmralign does not support images of different shapes",
    ):
        parcel_masker.fit(imgs)


def test_clustering_with_mask():
    """Test that ParcellationMasker raises ValueError if clustering is \
        provided with a bigger mask"""
    clustering_data = np.ones((8, 7, 6))
    clustering_data[5:, :, :] = 0
    clustering_img = Nifti1Image(clustering_data, np.eye(4))
    img, dummy_mask = random_niimg((8, 7, 6))
    parcel_masker = ParcellationMasker(
        clustering=clustering_img, mask=dummy_mask
    )
    with pytest.warns(
        UserWarning, match="Mask used was bigger than clustering provided"
    ):
        parcel_masker.fit(img)


def test_provided_masker():
    """Test that ParcellationMasker can use an existing masker"""
    img, mask = random_niimg((8, 7, 6))

    # Test with unfitted masker
    unfitted_masker = NiftiMasker(mask_img=mask)
    parcel_masker = ParcellationMasker(masker=unfitted_masker)
    parcel_masker.fit(img)

    assert hasattr(parcel_masker, "masker")
    assert parcel_masker.labels is not None

    # Test with fitted masker
    fitted_masker = NiftiMasker(mask_img=mask).fit(img)
    parcel_masker = ParcellationMasker(masker=fitted_masker)
    parcel_masker.fit(img)

    assert hasattr(parcel_masker, "masker")
    assert parcel_masker.labels is not None


def test_memory_caching(tmp_path):
    """Test that ParcellationMasker can use joblib memory caching"""
    img, _ = random_niimg((8, 7, 6))
    # Test that memory caching works
    memory = Memory(location=str(tmp_path), verbose=0)
    parcel_masker = ParcellationMasker(memory=memory, memory_level=1)
    parcel_masker.fit(img)

    # Check that cache directory is not empty
    cache_files = list(tmp_path.glob("joblib/*"))
    assert len(cache_files) > 0


@pytest.mark.parametrize("n_jobs", [1, 2, -1])
def test_parallel_processing(n_jobs):
    """Test parallel processing with joblib"""
    imgs = [random_niimg((8, 7, 6))[0]] * 3
    parcel_masker = ParcellationMasker(n_pieces=2, n_jobs=n_jobs)
    parcel_masker.fit(imgs)
    transformed_data = parcel_masker.transform(imgs)

    assert len(transformed_data) == 3


def test_smoothing_parameter():
    """Test that ParcellationMasker applies smoothing"""
    img, _ = random_niimg((8, 7, 6))
    parcel_masker = ParcellationMasker(smoothing_fwhm=4.0)
    parcel_masker.fit(img)
    transformed_data = parcel_masker.transform(img)

    assert isinstance(transformed_data, list)
    assert len(transformed_data) == 1


def test_standardization():
    """Test that ParcellationMasker standardizes data"""
    img, _ = random_niimg((8, 7, 6, 20))
    parcel_masker = ParcellationMasker(standardize=True)
    parcel_masker.fit(img)
    transformed_data = parcel_masker.transform(img)

    # Check if data is standardized (mean ≈ 0, std ≈ 1)
    data_array = transformed_data[0].data
    assert np.abs(np.mean(data_array)) < 1e-5
    assert np.abs(np.std(data_array) - 1.0) < 1e-5


def test_one_surface_image():
    """Test that ParcellationMasker can handle surface images"""
    img = surf_img(20)
    n_pieces = 2
    n_vertices_total = img.shape[0]
    parcel_masker = ParcellationMasker(n_pieces=n_pieces)
    parcel_masker.fit(img)

    assert hasattr(parcel_masker, "masker")
    assert parcel_masker.labels is not None
    assert isinstance(parcel_masker.labels, np.ndarray)
    assert len(np.unique(parcel_masker.labels)) == n_pieces
    assert len(parcel_masker.labels) == n_vertices_total


def test_multiple_surface_images():
    """Test that ParcellationMasker can handle multiple surface images"""
    imgs = [surf_img(20)] * 3
    n_pieces = 2
    n_vertices_total = imgs[0].shape[0]
    parcel_masker = ParcellationMasker(n_pieces=n_pieces)
    parcel_masker = parcel_masker.fit(imgs)

    assert hasattr(parcel_masker, "masker")
    assert parcel_masker.labels is not None
    assert isinstance(parcel_masker.labels, np.ndarray)
    assert len(np.unique(parcel_masker.labels)) == n_pieces
    assert len(parcel_masker.labels) == n_vertices_total


def test_one_contrast():
    """Test that ParcellationMasker handles both 3D and\n
    4D images in the case of one contrast"""
    img1, _ = random_niimg((8, 7, 6))
    img2, _ = random_niimg((8, 7, 6, 1))
    parcel_masker = ParcellationMasker()
    parcel_masker.fit([img1, img2])


def test_get_parcellation_img():
    """Test that ParcellationMasker returns the parcellation mask"""
    n_pieces = 2
    img, _ = random_niimg((8, 7, 6))
    parcel_masker = ParcellationMasker(n_pieces=n_pieces)
    parcel_masker.fit(img)
    parcellation_img = parcel_masker.get_parcellation_img()
    labels = parcel_masker.get_labels()

    assert isinstance(parcellation_img, Nifti1Image)
    assert parcellation_img.shape == img.shape

    masker = parcel_masker.masker
    data = masker.transform(parcellation_img)

    assert np.allclose(data, labels)
    assert len(np.unique(data)) == n_pieces


def test_clustering_surf():
    """Test that ParcellationMasker can use surface images as clustering"""
    img = surf_img(20)
    clustering_surf_img = surf_img(1)
    clustering_surf_img.data.parts["left"] = np.zeros((4, 1))
    clustering_surf_img.data.parts["right"] = np.ones((5, 1))

    parcel_masker = ParcellationMasker(clustering=clustering_surf_img)
    parcel_masker.fit(img)
    labels = parcel_masker.get_labels()

    assert np.allclose(labels, np.array(4 * [0] + 5 * [1]))
