import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from fmralign.alignment_methods import Identity
from fmralign.pairwise_alignment import PairwiseAlignment
from fmralign.tests.utils import assert_algo_transform_almost_exactly, random_nifti


def test_pairwise_identity():
    img1, mask_img = random_nifti((10, 10, 5, 5))
    identity = PairwiseAlignment(alignment_method='identity', mask=mask_img)
    assert_algo_transform_almost_exactly(identity, img1, img1, mask=mask_img)


def test_piecewise_identity():
    img1, mask_img = random_nifti((10, 10, 5, 5))
    identity = PairwiseAlignment(
        alignment_method='identity', n_pieces=3, mask=mask_img)
    assert_algo_transform_almost_exactly(identity, img1, img1, mask=mask_img)


def test_bagged_piecewise_identity():
    img1, mask_img = random_nifti((10, 10, 5, 5))
    identity = PairwiseAlignment(
        alignment_method='identity', n_pieces=3, n_bags=10, mask=mask_img)
    assert_algo_transform_almost_exactly(identity, img1, img1, mask=mask_img)


def test_bagged_piecewise_identity_2jobs():
    img1, mask_img = random_nifti((10, 10, 5, 5))
    identity = PairwiseAlignment(
        alignment_method='identity', n_pieces=3, n_bags=10, mask=mask_img, n_jobs=2)
    assert_algo_transform_almost_exactly(identity, img1, img1, mask=mask_img)


def test_hungarian_pairwise_alignment():

    X = np.array([[1., 4., 10], [1.5, 5, 10], [1, 5, 11], [1, 5.5, 8]])

    # translate the data matrix along features axis (voxels are permutated)
    Y = np.roll(X, 2, axis=1)
    from fmralign.alignment_methods import optimal_permutation, Hungarian

    assert_array_almost_equal(opt.dot(X.T).T, Y)

    hu = Hungarian()
    hu.fit(X, Y)
    assert_array_almost_equal(hu.transform(X), Y)

    X = np.array([[1., 4., 10], [1.5, 5, 10], [1, 5, 11], [1, 5.5, 8]])

    Y = np.roll(X, 2, axis=1)

    X = X[:, :, np.newaxis]
    img1_3d = nibabel.Nifti1Image(X, np.eye(4))
    img1_4d = nibabel.Nifti1Image(np.stack([X, X]), np.eye(4))
    # translate the data matrix along one axis
    Y = Y[:, :, np.newaxis]
    img2_3d = nibabel.Nifti1Image(Y, np.eye(4))
    img2_4d = nibabel.Nifti1Image(np.stack([Y, Y]), np.eye(4))

    mask_img = nibabel.Nifti1Image(np.ones(X.shape, dtype=np.int8), np.eye(4))

    # With mask :
    permutation_with_mask = PairwiseAlignment(
        alignment_method='permutation', mask=mask_img)
    permutation_with_mask.fit(img1_3d, img2_3d)
    permutation_with_mask.fit_[0][0].R.toarray().shape
    assert_algo_transform_almost_exactly(
        permutation_with_mask, img1_3d, img2_3d, mask=mask_img)

    # without mask :
    permutation_without_mask = PairwiseAlignment(
        alignment_method='permutation')
    assert_algo_transform_almost_exactly(
        permutation_without_mask, img1_4d, img2_4d)


def test_pairwise_ridge():

    img1, mask_img = random_nifti((10, 10, 5, 5))
    img2, _ = random_nifti((10, 10, 5, 5))

    pairwise_ridge = PairwiseAlignment(
        alignment_method='ridge_cv', mask=mask_img)

    assert_model_align_better_than_identity(
        pairwise_ridge, img1, img2, mask_img)
