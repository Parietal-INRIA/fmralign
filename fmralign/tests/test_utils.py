# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from nilearn.input_data import NiftiMasker
import nibabel
import pytest
from fmralign.tests.utils import random_niimg
from fmralign._utils import _make_parcellation, _hierarchical_k_means
import nilearn
from packaging import version
from nilearn._utils.data_gen import generate_fake_fmri
from itertools import product


def test_hierarchical_k_means():
    X = [[10, -10, 30], [12, -8, 24]]
    truth_labels = np.tile([0, 1, 2], 5)
    X = np.tile(X, 5).T
    test_labels = _hierarchical_k_means(X, 3)
    truth_labels = np.tile([test_labels[0], test_labels[1], test_labels[2]], 5)
    assert_array_almost_equal(test_labels, truth_labels)


def test_make_parcellation():
    # make_parcellation is built on Nilearn which already has several test for its Parcellation class
    # here we test just the call of the API is right on a simple example
    img, mask_img = random_niimg((7, 6, 8, 5))
    indexes = np.arange(img.shape[-1])
    masker = NiftiMasker(mask_img=mask_img).fit()

    methods = ["kmeans", "ward", "hierarchical_kmeans"]

    # check rena only if nilearn version allow it
    if version.parse(nilearn.__version__) <= version.parse("0.5.2"):
        with pytest.raises(Exception):
            assert _make_parcellation(img, "rena", n_pieces, masker)
    else:
        methods.append("rena")

    for clustering_method in methods:
        # check n_pieces = 1 gives out ones of right shape
        assert (_make_parcellation(
            img, indexes, clustering_method, 1, masker) == masker.transform(mask_img)).all()

        # check n_pieces = 2 find right clustering
        labels = _make_parcellation(
            img, indexes, clustering_method, 2, masker)
        assert(len(np.unique(labels)) == 2)

        # check that not inputing n_pieces yields problems
        with pytest.raises(Exception):
            assert _make_parcellation(
                img, indexes, clustering_method, 0, masker)

    clustering = nibabel.Nifti1Image(
        np.hstack([np.ones((7, 3, 8)), 2 * np.ones((7, 3, 8))]), np.eye(4))

    # check 3D Niimg clusterings
    for n_pieces in [0, 1, 2]:
        labels = _make_parcellation(
            img, indexes, clustering, n_pieces, masker)
        assert(len(np.unique(labels)) == 2)

    # check warning if a parcel is too big
    with pytest.warns(UserWarning):
        clustering = nibabel.Nifti1Image(
            np.hstack([np.ones(2000), 4 * np.ones(800)]), np.eye(4))
        _make_parcellation(
            img, indexes, clustering_method, n_pieces, masker)
