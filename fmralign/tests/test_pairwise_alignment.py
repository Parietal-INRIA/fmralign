import numpy as np
import nibabel
from sklearn.utils.testing import assert_array_almost_equal
from fmralign.pairwise_alignment import PairwiseAlignment
from fmralign.tests.utils import assert_algo_transform_almost_exactly, random_niimg, assert_model_align_better_than_identity
from fmralign.alignment_methods import optimal_permutation, Hungarian


def test_pairwise_identity():
    img1, mask_img = random_niimg((10, 10, 5, 5))

    args_list = [{'alignment_method': 'identity', 'mask': mask_img},
                 {'alignment_method': 'identity', 'n_pieces': 3, 'mask': mask_img},
                 {'alignment_method': 'identity', 'n_pieces': 3,
                     'n_bags': 10, 'mask': mask_img},
                 {'alignment_method': 'identity', 'n_pieces': 3,
                     'n_bags': 10, 'mask': mask_img, 'n_jobs': 2}
                 ]
    for args in args_list:
        algo = PairwiseAlignment(**args)
        assert_algo_transform_almost_exactly(
            algo, img1, img1, mask=mask_img)


def test_models_against_identity():
    img1, mask_img = random_niimg((14, 12, 10, 5))
    img2, _ = random_niimg((14, 12, 10, 5))
    import numpy as np
    algo = PairwiseAlignment(
        alignment_method="optimal_transport", mask=mask_img, n_pieces=2, n_bags=1, n_jobs=1)
    algo.fit(img1, img2)
    for alignment_method in ['optimal_transport', 'scaled_orthogonal', 'permutation',  'ridge_cv']:
        algo = PairwiseAlignment(
            alignment_method=alignment_method, mask=mask_img, n_pieces=2, n_bags=1, n_jobs=1)
        algo.fit(img1, img2)
        print(algo.labels_[0].shape)
        print(algo.fit_[0][0].R.shape)
        print(algo.fit_[0][1].R.shape)
    algo.labels_.shape
    algo.fit_[0][1].R
    algo.fit_[0][0].R.shape
    algo.labels_[0].shape
    for label in np.unique(algo.labels_[0]):
        X_transform[labels == i] = estimators[i].transform(X[labels == i])

    im_test = algo.transform(img1)
    for alignment_method in ['scaled_orthogonal', 'permutation',  'ridge_cv', 'optimal_transport']:
        algo = PairwiseAlignment(
            alignment_method=alignment_method, mask=mask_img, n_pieces=2, n_bags=1, n_jobs=1)
        print(alignment_method)
        assert_model_align_better_than_identity(
            algo, img1, img2, mask_img)


def test_hungarian_pairwise_alignment():

    X = np.array([[1., 4., 10], [1.5, 5, 10], [1, 5, 11], [1, 5.5, 8]])

    # translate the data matrix along features axis (voxels are permutated)
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
