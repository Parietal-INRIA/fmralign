import numpy as np
import pytest
from joblib import Memory
from nibabel.nifti1 import Nifti1Image

from fmralign._utils import ParceledData
from fmralign.preprocessing import ParcellationMasker
from fmralign.tests.utils import random_niimg


def test_init_default_params():
    """Test that ParcellationMasker initializes with default parameters"""
    pmasker = ParcellationMasker()
    assert pmasker.n_pieces == 1
    assert pmasker.clustering == "kmeans"
    assert pmasker.mask is None
    assert pmasker.smoothing_fwhm is None
    assert pmasker.standardize is False
    assert pmasker.detrend is False
    assert pmasker.labels is None


def test_init_custom_params():
    """Test that ParcellationMasker initializes with custom parameters"""
    pmasker = ParcellationMasker(
        n_pieces=2, clustering="ward", standardize=True, detrend=True, n_jobs=2
    )
    assert pmasker.n_pieces == 2
    assert pmasker.clustering == "ward"
    assert pmasker.standardize is True
    assert pmasker.detrend is True
    assert pmasker.n_jobs == 2


def test_fit_single_image():
    """Test that ParcellationMasker fits a single image"""
    img, _ = random_niimg((8, 7, 6))
    pmasker = ParcellationMasker(n_pieces=2)
    fitted_pmasker = pmasker.fit(img)

    assert hasattr(fitted_pmasker, "masker_")
    assert fitted_pmasker.labels is not None
    assert isinstance(fitted_pmasker.labels, np.ndarray)
    assert len(np.unique(fitted_pmasker.labels)) == 2  # n_pieces=2


def test_fit_multiple_images():
    """Test that ParcellationMasker fits multiple images"""
    imgs = [random_niimg((8, 7, 6))[0]] * 3
    pmasker = ParcellationMasker(n_pieces=2)
    fitted_pmasker = pmasker.fit(imgs)

    assert hasattr(fitted_pmasker, "masker_")
    assert fitted_pmasker.labels is not None


def test_transform_single_image():
    """Test that ParcellationMasker transforms a single image"""
    img, _ = random_niimg((8, 7, 6))
    pmasker = ParcellationMasker(n_pieces=2)
    pmasker.fit(img)
    transformed_data = pmasker.transform(img)

    assert isinstance(transformed_data, list)
    assert len(transformed_data) == 1
    assert isinstance(transformed_data[0], ParceledData)


def test_transform_multiple_images():
    """Test that ParcellationMasker transforms multiple images"""
    imgs = [random_niimg((8, 7, 6))[0]] * 3
    pmasker = ParcellationMasker(n_pieces=2)
    pmasker.fit(imgs)
    transformed_data = pmasker.transform(imgs)

    assert isinstance(transformed_data, list)
    assert len(transformed_data) == 3
    assert all(
        isinstance(parceled_data, ParceledData)
        for parceled_data in transformed_data
    )


def test_get_labels_before_fit():
    """Test that ParcellationMasker raises ValueError if get_labels is called before fit"""
    pmasker = ParcellationMasker()
    with pytest.raises(ValueError, match="Labels have not been computed yet"):
        pmasker.get_labels()


def test_get_labels_after_fit():
    img, _ = random_niimg((8, 7, 6))
    pmasker = ParcellationMasker(n_pieces=2)
    pmasker.fit(img)
    labels = pmasker.get_labels()

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
    pmasker = ParcellationMasker()

    with pytest.raises(
        NotImplementedError,
        match="fmralign does not support images of different shapes",
    ):
        pmasker.fit(imgs)


def test_clustering_with_mask():
    """Test that ParcellationMasker raises ValueError if clustering is \
        provided with a bigger mask"""
    clustering_data = np.ones((8, 7, 6))
    clustering_data[5:, :, :] = 0
    clustering_img = Nifti1Image(clustering_data, np.eye(4))
    img, dummy_mask = random_niimg((8, 7, 6))
    pmasker = ParcellationMasker(clustering=clustering_img, mask=dummy_mask)
    with pytest.warns(
        UserWarning, match="Mask used was bigger than clustering provided"
    ):
        pmasker.fit(img)


def test_memory_caching(tmp_path):
    """Test that ParcellationMasker can use joblib memory caching"""
    img, _ = random_niimg((8, 7, 6))
    # Test that memory caching works
    memory = Memory(location=str(tmp_path), verbose=0)
    pmasker = ParcellationMasker(memory=memory, memory_level=1)
    pmasker.fit(img)

    # Check that cache directory is not empty
    cache_files = list(tmp_path.glob("joblib/*"))
    assert len(cache_files) > 0


@pytest.mark.parametrize("n_jobs", [1, 2, -1])
def test_parallel_processing(n_jobs):
    """Test parallel processing with joblib"""
    imgs = [random_niimg((8, 7, 6))[0]] * 3
    pmasker = ParcellationMasker(n_pieces=2, n_jobs=n_jobs)
    pmasker.fit(imgs)
    transformed_data = pmasker.transform(imgs)

    assert len(transformed_data) == 3


def test_smoothing_parameter():
    """Test that ParcellationMasker applies smoothing"""
    img, _ = random_niimg((8, 7, 6))
    pmasker = ParcellationMasker(smoothing_fwhm=4.0)
    pmasker.fit(img)
    transformed_data = pmasker.transform(img)

    assert isinstance(transformed_data, list)
    assert len(transformed_data) == 1


def test_standardization():
    """Test that ParcellationMasker standardizes data"""
    img, _ = random_niimg((8, 7, 6, 20))
    pmasker = ParcellationMasker(standardize=True)
    pmasker.fit(img)
    transformed_data = pmasker.transform(img)

    # Check if data is standardized (mean ≈ 0, std ≈ 1)
    data_array = transformed_data[0].data
    assert np.abs(np.mean(data_array)) < 1e-5
    assert np.abs(np.std(data_array) - 1.0) < 1e-5


def test_get_parcellation():
    """Test that ParcellationMasker returns the parcellation mask"""
    n_pieces = 2
    img, _ = random_niimg((8, 7, 6))
    pmasker = ParcellationMasker(n_pieces=n_pieces)
    pmasker.fit(img)
    parcellation_img = pmasker.get_parcellation()
    labels = pmasker.get_labels()

    assert isinstance(parcellation_img, Nifti1Image)
    assert parcellation_img.shape == img.shape

    masker = pmasker.masker_
    data = masker.transform(parcellation_img)

    assert np.allclose(data, labels)
    assert len(np.unique(data)) == n_pieces


if __name__ == "__main__":
    test_get_parcellation()
