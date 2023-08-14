# -*- coding: utf-8 -*-
import nibabel
import numpy as np
import pytest
from nilearn.maskers import NiftiMasker

from fmralign._utils import _make_parcellation
from fmralign.tests.utils import random_niimg


def test_make_parcellation():
    # make_parcellation is built on Nilearn which already has several test for its Parcellation class
    # here we test just the call of the API is right on a simple example
    img, mask_img = random_niimg((7, 6, 8, 5))
    indexes = np.arange(img.shape[-1])
    masker = NiftiMasker(mask_img=mask_img).fit()

    methods = ["kmeans", "ward", "hierarchical_kmeans", "rena"]

    for clustering_method in methods:
        # check n_pieces = 1 gives out ones of right shape
        assert (
            _make_parcellation(img, indexes, clustering_method, 1, masker)
            == masker.transform(mask_img)
        ).all()

        # check n_pieces = 2 find right clustering
        labels = _make_parcellation(img, indexes, clustering_method, 2, masker)
        assert len(np.unique(labels)) == 2

        # check that not inputing n_pieces yields problems
        with pytest.raises(Exception):
            assert _make_parcellation(img, indexes, clustering_method, 0, masker)

    clustering = nibabel.Nifti1Image(
        np.hstack([np.ones((7, 3, 8)), 2 * np.ones((7, 3, 8))]), np.eye(4)
    )

    # check 3D Niimg clusterings
    for n_pieces in [0, 1, 2]:
        labels = _make_parcellation(img, indexes, clustering, n_pieces, masker)
        assert len(np.unique(labels)) == 2

    # check warning if a parcel is too big
    with pytest.warns(UserWarning):
        clustering = nibabel.Nifti1Image(
            np.hstack([np.ones(2000), 4 * np.ones(800)]), np.eye(4)
        )
        _make_parcellation(img, indexes, clustering_method, n_pieces, masker)
