import pytest
import numpy as np
from nibabel.nifti1 import Nifti1Image
from nilearn._utils.class_inspect import check_estimator
from joblib import Memory

from fmralign.preprocessing import Preprocessor
from fmralign.tests.utils import random_niimg
from fmralign._utils import ParceledData


def test_init_default_params():
    preprocessor = Preprocessor()
    assert preprocessor.n_pieces == 1
    assert preprocessor.clustering == "kmeans"
    assert preprocessor.mask is None
    assert preprocessor.smoothing_fwhm is None
    assert preprocessor.standardize is False
    assert preprocessor.detrend is False
    assert preprocessor.labels is None


def test_init_custom_params():
    preprocessor = Preprocessor(
        n_pieces=2, clustering="ward", standardize=True, detrend=True, n_jobs=2
    )
    assert preprocessor.n_pieces == 2
    assert preprocessor.clustering == "ward"
    assert preprocessor.standardize is True
    assert preprocessor.detrend is True
    assert preprocessor.n_jobs == 2


def test_fit_single_image():
    img, _ = random_niimg((8, 7, 6))
    preprocessor = Preprocessor(n_pieces=2)
    fitted_preprocessor = preprocessor.fit(img)

    assert hasattr(fitted_preprocessor, "masker_")
    assert fitted_preprocessor.labels is not None
    assert isinstance(fitted_preprocessor.labels, np.ndarray)
    assert len(np.unique(fitted_preprocessor.labels)) == 2  # n_pieces=2


def test_fit_multiple_images():
    imgs = [random_niimg((8, 7, 6))[0]] * 3
    preprocessor = Preprocessor(n_pieces=2)
    fitted_preprocessor = preprocessor.fit(imgs)

    assert hasattr(fitted_preprocessor, "masker_")
    assert fitted_preprocessor.labels is not None


def test_transform_single_image():
    img, _ = random_niimg((8, 7, 6))
    preprocessor = Preprocessor(n_pieces=2)
    preprocessor.fit(img)
    transformed_data = preprocessor.transform(img)

    assert isinstance(transformed_data, list)
    assert len(transformed_data) == 1
    assert isinstance(transformed_data[0], ParceledData)


def test_transform_multiple_images():
    imgs = [random_niimg((8, 7, 6))[0]] * 3
    preprocessor = Preprocessor(n_pieces=2)
    preprocessor.fit(imgs)
    transformed_data = preprocessor.transform(imgs)

    assert isinstance(transformed_data, list)
    assert len(transformed_data) == 3
    assert all(
        isinstance(parceled_data, ParceledData)
        for parceled_data in transformed_data
    )


def test_get_labels_before_fit():
    preprocessor = Preprocessor()
    with pytest.raises(ValueError, match="Labels have not been computed yet"):
        preprocessor.get_labels()


def test_get_labels_after_fit():
    img, _ = random_niimg((8, 7, 6))
    preprocessor = Preprocessor(n_pieces=2)
    preprocessor.fit(img)
    labels = preprocessor.get_labels()

    assert labels is not None
    assert isinstance(labels, np.ndarray)
    assert len(np.unique(labels)) == 2


def test_different_shaped_images():
    img, _ = random_niimg((8, 7, 6))
    # Create image with different shape
    different_data = np.random.rand(8, 7, 8)
    different_img = Nifti1Image(different_data, np.eye(4))

    imgs = [img, different_img]
    preprocessor = Preprocessor()

    with pytest.raises(
        NotImplementedError,
        match="fmralign does not support images of different shapes",
    ):
        preprocessor.fit(imgs)


def test_clustering_with_mask():
    clustering_data = np.ones((8, 7, 6))
    clustering_data[5:, :, :] = 0
    clustering_img = Nifti1Image(clustering_data, np.eye(4))
    img, dummy_mask = random_niimg((8, 7, 6))
    preprocessor = Preprocessor(clustering=clustering_img, mask=dummy_mask)
    with pytest.warns(
        UserWarning, match="Mask used was bigger than clustering provided"
    ):
        preprocessor.fit(img)


def test_memory_caching(tmp_path):
    img, _ = random_niimg((8, 7, 6))
    # Test that memory caching works
    memory = Memory(location=str(tmp_path), verbose=0)
    preprocessor = Preprocessor(memory=memory, memory_level=1)
    preprocessor.fit(img)

    # Check that cache directory is not empty
    cache_files = list(tmp_path.glob("joblib/*"))
    assert len(cache_files) > 0


@pytest.mark.parametrize("n_jobs", [1, 2, -1])
def test_parallel_processing(n_jobs):
    imgs = [random_niimg((8, 7, 6))[0]] * 3
    preprocessor = Preprocessor(n_pieces=2, n_jobs=n_jobs)
    preprocessor.fit(imgs)
    transformed_data = preprocessor.transform(imgs)

    assert len(transformed_data) == 3


def test_scikit_learn_compatibility():
    # Test if the estimator adheres to scikit-learn conventions
    check_estimator(Preprocessor())


def test_smoothing_parameter():
    img, _ = random_niimg((8, 7, 6))
    preprocessor = Preprocessor(smoothing_fwhm=4.0)
    preprocessor.fit(img)
    transformed_data = preprocessor.transform(img)

    assert isinstance(transformed_data, list)
    assert len(transformed_data) == 1


def test_standardization():
    img, _ = random_niimg((8, 7, 6, 20))
    preprocessor = Preprocessor(standardize=True)
    preprocessor.fit(img)
    transformed_data = preprocessor.transform(img)

    # Check if data is standardized (mean ≈ 0, std ≈ 1)
    data_array = transformed_data[0].data
    assert np.abs(np.mean(data_array)) < 1e-5
    assert np.abs(np.std(data_array) - 1.0) < 1e-5


if __name__ == "__main__":
    test_scikit_learn_compatibility()
