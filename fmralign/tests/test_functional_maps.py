import pytest
from nilearn.maskers import SurfaceMasker
from numpy.testing import assert_array_almost_equal

from fmralign.functional_maps import FMAPAlignment
from fmralign.tests.utils import surf_img


@pytest.mark.parametrize("backend", ["pot", "keops"])
def test_identity_alignment(backend):
    """Test that alpha=0.0 leads to identity alignment"""
    img1 = surf_img(20)
    img2 = surf_img(20)

    masker = SurfaceMasker().fit([img1, img2])

    algo = FMAPAlignment(
        masker=masker,
        n_lb=3,
        alpha=0.0,
        n_iter=100,
        reg=1e-6,
        backend=backend,
    )
    algo.fit(img1, img2)
    img1_aligned = algo.transform(img1)

    assert_array_almost_equal(
        masker.transform(img1),
        masker.transform(img1_aligned),
    )


@pytest.mark.parametrize("reg", [1e-6, 1e-3, 1e-1])
def test_backend_invariance(reg):
    """Test that the results are the same regardless of the backend"""
    img1 = surf_img(20)
    img2 = surf_img(20)

    masker = SurfaceMasker().fit([img1, img2])

    algo1 = FMAPAlignment(
        masker=masker,
        n_lb=3,
        reg=reg,
        backend="pot",
    )
    algo1.fit(img1, img2)
    pot_transform = algo1.transform(img1)

    algo2 = FMAPAlignment(
        masker=masker,
        n_lb=3,
        reg=reg,
        backend="keops",
    )
    algo2.fit(img1, img2)
    keops_transform = algo2.transform(img1)

    assert_array_almost_equal(
        masker.transform(pot_transform),
        masker.transform(keops_transform),
    )
