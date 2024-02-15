import pytest

import numpy as np
from sklearn.base import clone
from nilearn.maskers import NiftiMasker
from fastsrm.identifiable_srm import IdentifiableFastSRM

from fmralign.srm import PiecewiseModel, Identity


def to_niimgs(X, dim):
    from nilearn.masking import unmask_from_to_3d_array
    import nibabel

    p = np.prod(dim)
    assert len(dim) == 3
    assert X.shape[-1] <= p
    mask = np.zeros(p).astype(bool)
    mask[: X.shape[-1]] = 1
    assert mask.sum() == X.shape[1]
    mask = mask.reshape(dim)
    X = np.rollaxis(np.array([unmask_from_to_3d_array(x, mask) for x in X]), 0, start=4)
    affine = np.eye(4)
    return (
        nibabel.Nifti1Image(X, affine),
        nibabel.Nifti1Image(mask.astype(float), affine),
    )


n_bags = 1
n_pieces = 1
n_timeframes_align = 1000
n_timeframes_test = 200
n_subjects = 3
# n_voxels should have an integer cubic root (8 ; 27 ; 64 ; 125 ...)
n_voxels = 8

grid_l = int(pow(n_voxels, 1.0 / 3.0))

_, mask = to_niimgs(
    np.random.rand(n_timeframes_align, n_voxels), (grid_l, grid_l, grid_l)
)
masker = NiftiMasker(mask_img=mask).fit()

train_data, test_data = {}, {}
for i in range(n_subjects):
    train_data[i] = masker.inverse_transform(np.random.rand(n_timeframes_align, 8))
    test_data[i] = masker.inverse_transform(np.random.rand(n_timeframes_test, 8))
masker = NiftiMasker(mask_img=mask).fit()

n_components = 3


srm = IdentifiableFastSRM(n_components=n_components, aggregate=None)
algos_to_test = [
    # "identity",
    IdentifiableFastSRM(n_components=n_components, aggregate=None),
]


@pytest.mark.parametrize("algo", algos_to_test)
def test_output_no_clustering(algo):
    if algo == "identity":
        n_comp = n_voxels
        psrm = PiecewiseModel(
            "identity",
            n_pieces=n_pieces,
            n_bags=n_bags,
            clustering="kmeans",
            mask=masker,
            n_jobs=-1,
        )
        algo = Identity()
    else:
        n_comp = algo.n_components
        psrm = PiecewiseModel(
            algo,
            n_pieces=n_pieces,
            n_bags=n_bags,
            clustering="kmeans",
            mask=masker,
            n_jobs=-1,
        )
    psrm.fit(list(train_data.values())[:-1])

    srm_SR = algo.fit_transform(
        [masker.transform(x).T for x in list(train_data.values())[:-1]]
    )
    if len(srm_SR) == (n_subjects - 1):
        srm_SR = np.mean(srm_SR, axis=0)

    np.shape(psrm.reduced_sr)
    assert np.shape(psrm.reduced_sr) == (
        n_bags,
        n_pieces,
        n_comp,
        n_timeframes_align,
    )
    assert np.shape(psrm.labels_) == (n_bags, n_voxels)
    assert np.shape(psrm.fit_) == (n_bags, n_pieces)
    np.testing.assert_almost_equal(psrm.reduced_sr[0][0], srm_SR)

    algo.add_subjects([masker.transform(list(train_data.values())[-1]).T], srm_SR)
    psrm.add_subjects([list(train_data.values())[-1]])

    np.testing.assert_almost_equal(psrm.reduced_sr[0][0], srm_SR)
    np.testing.assert_almost_equal(psrm.fit_[0][0].basis_list, algo.basis_list)

    aligned_test = psrm.transform(test_data.values())
    if hasattr(algo, "aggregate"):
        algo.aggregate = None
    srm_aligned_test = algo.transform(
        [masker.transform(y).T for y in list(test_data.values())]
    )
    assert np.shape(aligned_test) == (n_subjects, n_comp, n_timeframes_test)
    np.testing.assert_almost_equal(aligned_test, srm_aligned_test)


def test_identity():
    id = Identity()
    id_SR = id.fit_transform(
        [masker.transform(x).T for x in list(train_data.values())[:-1]]
    )
    id.add_subjects([masker.transform(list(train_data.values())[-1]).T], id_SR)
    _ = id.transform([masker.transform(y).T for y in list(test_data.values())])
    # Check identity SRM just returns the data
    np.testing.assert_almost_equal(
        id_SR.shape,
        np.asarray(
            [masker.transform(x).T for x in list(train_data.values())[:-1]]
        ).shape,
    )


@pytest.mark.parametrize("algo", algos_to_test)
def test_algo_each_piece(algo):
    # Test that doing stuff piecewise give the same
    # result as doing it separately
    X = np.random.rand(1000, 8)
    X1, mask = to_niimgs(X, (2, 2, 2))
    masker = NiftiMasker(mask_img=mask).fit()
    X2 = masker.inverse_transform(np.random.rand(1000, 8))
    X3 = masker.inverse_transform(np.random.rand(1000, 8))
    X4 = masker.inverse_transform(np.random.rand(1000, 8))

    cluster = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    niimg_cluster = masker.inverse_transform(cluster)
    if algo == "identity":
        srm = PiecewiseModel("identity", mask=masker, clustering=niimg_cluster)
        algo = Identity()
    else:
        srm = PiecewiseModel(algo, clustering=niimg_cluster, mask=masker)
    S1 = np.array(
        [
            clone(algo).fit_transform(
                [masker.transform(x).T[cluster == i] for x in [X1, X2, X3, X4]]
            )
            for i in [1, 2]
        ]
    )

    S2 = srm.fit([X1, X2, X3, X4]).transform([X1, X2, X3, X4])
    S1 = np.swapaxes(S1, 0, 1)
    S1 = S1.reshape(4, -1, 1000)
    np.testing.assert_almost_equal(S1, S2)


n_components = 4


@pytest.mark.parametrize("algo", algos_to_test)
def test_wrongshape(algo):
    # Test that doing stuff piecewise give the same
    # result as doing it separately
    X = np.random.rand(10, 64)
    X1, mask = to_niimgs(X, (4, 4, 4))
    masker = NiftiMasker(mask_img=mask).fit()
    X2 = masker.inverse_transform(np.random.rand(10, 64))
    X3 = masker.inverse_transform(np.random.rand(10, 64))
    X4 = masker.inverse_transform(np.random.rand(10, 64))

    cluster = np.array([1] * 8 + [2] * 56)
    niimg_cluster = masker.inverse_transform(cluster)
    if algo == "identity":
        srm = PiecewiseModel("identity", mask=masker, clustering=niimg_cluster)
        algo = Identity()
    else:
        srm = PiecewiseModel(algo, clustering=niimg_cluster, mask=masker)
        S1 = np.array(
            [
                clone(algo).fit_transform(
                    [masker.transform(x).T[cluster == i] for x in [X1, X2, X3, X4]]
                )
                for i in [1, 2]
            ]
        )

        S2 = srm.fit([X1, X2, X3, X4]).transform([X1, X2, X3, X4])
        S1 = np.swapaxes(S1, 0, 1)
        S1 = S1.reshape(4, -1, 10)
        np.testing.assert_almost_equal(S1, S2)
