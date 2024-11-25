# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
import pytest
from nilearn.maskers import NiftiMasker
from numpy.testing import assert_array_almost_equal, assert_array_equal

from fmralign._utils import ParceledData, _make_parcellation
from fmralign.tests.utils import random_niimg, sample_parceled_data


def test_make_parcellation():
    # make_parcellation is built on Nilearn which already
    # has several test for its Parcellation class
    # here we test just the call of the API is right on a simple example
    img, mask_img = random_niimg((7, 6, 8, 5))
    masker = NiftiMasker(mask_img=mask_img).fit()

    methods = ["kmeans", "ward", "hierarchical_kmeans", "rena"]

    for clustering_method in methods:
        # check n_pieces = 1 gives out ones of right shape
        assert (
            _make_parcellation(img, clustering_method, 1, masker)
            == masker.transform(mask_img)
        ).all()

        # check n_pieces = 2 find right clustering
        labels = _make_parcellation(img, clustering_method, 2, masker)
        assert len(np.unique(labels)) == 2

        # check that not inputing n_pieces yields problems
        with pytest.raises(Exception):
            assert _make_parcellation(img, clustering_method, 0, masker)

    clustering = nib.Nifti1Image(
        np.hstack([np.ones((7, 3, 8)), 2 * np.ones((7, 3, 8))]), np.eye(4)
    )

    # check 3D Niimg clusterings
    for n_pieces in [0, 1, 2]:
        labels = _make_parcellation(img, clustering, n_pieces, masker)
        assert len(np.unique(labels)) == 2

    # check warning if a parcel is too big
    with pytest.warns(UserWarning):
        clustering = nib.Nifti1Image(
            np.hstack([np.ones(2000), 4 * np.ones(800)]), np.eye(4)
        )
        _make_parcellation(img, clustering_method, n_pieces, masker)


def test_initialization():
    """Test initialization of ParceledData class."""
    data, masker, labels = sample_parceled_data(n_pieces=2)
    parceled = ParceledData(data, masker, labels)

    assert_array_equal(parceled.data, data)
    assert parceled.masker == masker
    assert_array_equal(parceled.labels, labels)
    assert_array_equal(parceled.unique_labels, [1, 2])
    assert parceled.n_pieces == 2


def test_getitem_single_parcel():
    data, masker, labels = sample_parceled_data(n_pieces=2)
    parceled = ParceledData(data, masker, labels)

    # Test getting every parcel individually
    for i in range(2):
        parcel = parceled[i]
        expected = data[:, labels == i + 1]
        assert_array_equal(parcel, expected)


def test_getitem_slice():
    data, masker, labels = sample_parceled_data(n_pieces=2)
    parceled = ParceledData(data, masker, labels)

    # Test getting all parcels with slice
    all_parcels = parceled[:]
    assert len(all_parcels) == 2
    assert_array_equal(all_parcels[0], parceled[0])
    assert_array_equal(all_parcels[1], parceled[1])

    # Test step parameter
    stepped_parcels = parceled[::2]
    assert len(stepped_parcels) == 1
    assert_array_equal(stepped_parcels[0], parceled[0])


def test_getitem_invalid_key():
    data, masker, labels = sample_parceled_data(n_pieces=2)
    parceled = ParceledData(data, masker, labels)

    with pytest.raises(ValueError, match="Invalid key type."):
        parceled["invalid"]


def test_get_parcel():
    data, masker, labels = sample_parceled_data(n_pieces=2)
    parceled = ParceledData(data, masker, labels)

    # Test getting parcel with label 1
    parcel_0 = parceled.get_parcel(1)
    expected = data[:, labels == 1]
    assert_array_equal(parcel_0, expected)

    # Test getting parcel with label 2
    parcel_1 = parceled.get_parcel(2)
    expected = data[:, labels == 2]
    assert_array_equal(parcel_1, expected)


def test_to_list():
    data, masker, labels = sample_parceled_data(n_pieces=2)
    parceled = ParceledData(data, masker, labels)

    parcel_list = parceled.to_list()
    assert len(parcel_list) == 2
    assert_array_equal(parcel_list[0], parceled[0])
    assert_array_equal(parcel_list[1], parceled[1])


def test_to_img():
    data, masker, labels = sample_parceled_data(n_pieces=2)
    parceled = ParceledData(data, masker, labels)

    img = parceled.to_img()
    assert isinstance(img, nib.Nifti1Image)

    # Test shape of the reconstructed image
    assert img.shape == (8, 7, 6, 20)

    # Test that we can transform back to original data
    reconstructed_data = masker.transform(img)
    assert_array_almost_equal(reconstructed_data, data)


def test_edge_cases():
    # Test with single parcel
    data, masker, labels = sample_parceled_data(n_pieces=1)
    parceled = ParceledData(data, masker, labels)
    assert parceled.n_pieces == 1
    assert_array_equal(parceled[0], data)
    assert 0 == 0


def test_out_of_bounds_access():
    data, masker, labels = sample_parceled_data(n_pieces=2)
    parceled = ParceledData(data, masker, labels)

    with pytest.raises(IndexError):
        parceled[2]  # Trying to access non-existent parcel

    with pytest.raises(IndexError):
        parceled[-3]  # Trying to access with invalid negative index


def test_slice_bounds():
    data, masker, labels = sample_parceled_data(n_pieces=2)
    parceled = ParceledData(data, masker, labels)

    # # Test slices beyond bounds
    # with pytest.raises(IndexError):
    #     parceled[2:10]

    # Test negative slices
    result = parceled[-1:-3:-1]
    assert len(result) == 2
    assert_array_equal(result[1], parceled[0])
    assert_array_equal(result[0], parceled[1])


def test_non_contiguous_labels():
    data, masker, labels = sample_parceled_data(n_pieces=2)
    # Replace labels 1, 2 with 1, 3
    labels[labels == 2] = 3
    parceled = ParceledData(data, masker, labels)

    assert_array_equal(parceled.unique_labels, [1, 3])
    assert parceled.n_pieces == 2

    # Test accessing by index
    first_parcel = parceled[0]
    expected = data[:, labels == 1]
    assert_array_equal(first_parcel, expected)

    # Test accessing by label
    same_parcel = parceled.get_parcel(1)
    assert_array_equal(same_parcel, expected)


if __name__ == "__main__":
    test_non_contiguous_labels()
