import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from fmralign.alignment_methods import scaled_procrustes, ScaledOrthogonalAlignment
from fmralign.pairwise_alignment import PairwiseAlignment
from scipy.linalg import orthogonal_procrustes
from fmralign.tests.utils import assert_algo_transform_almost_exactly
import nibabel


def test_procrustes_null_input():
    X = np.random.randn(10, 20)
    Y = np.zeros_like(X)
    R = np.eye(X.shape[1])
    R_test, _ = scaled_procrustes(X, Y)
    assert_array_almost_equal(R, R_test.toarray())


def test_scaled_procrustes_primal_dual():
    n, p = 100, 20
    X = np.random.randn(n, p)
    Y = np.random.randn(n, p)
    R1, s1 = scaled_procrustes(X, Y, scaling=True, primal=True)
    R2, s2 = scaled_procrustes(X, Y, scaling=True, primal=False)
    assert_array_almost_equal(R1, R2)
    n, p = 20, 100
    X = np.random.randn(n, p)
    Y = np.random.randn(n, p)
    R1, s1 = scaled_procrustes(X, Y, scaling=True, primal=True)
    R2, s2 = scaled_procrustes(X, Y, scaling=True, primal=False)
    assert_array_almost_equal(R1.dot(X.T), R2.dot(X.T))


def test_scaled_procrustes_3Drotation():
    R = np.array([[1., 0., 0.], [0., np.cos(1), -np.sin(1)],
                  [0., np.sin(1), np.cos(1)]])
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)

    R_test, _ = scaled_procrustes(X.T, Y.T)
    assert_array_almost_equal(
        R.dot(np.array([0., 1., 0.])),
        np.array([0., np.cos(1), np.sin(1)])
    )
    assert_array_almost_equal(
        R.dot(np.array([0., 0., 1.])),
        np.array([0., -np.sin(1), np.cos(1)])
    )
    assert_array_almost_equal(R, R_test)


def test_Scaled_Orthogonal_Alignment_3Drotation():
    R = np.array([[1., 0., 0.], [0., np.cos(1), -np.sin(1)],
                  [0., np.sin(1), np.cos(1)]])
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)

    ortho_al = ScaledOrthogonalAlignment(scaling=False)
    ortho_al.fit(X.T, Y.T)

    assert_array_almost_equal(
        ortho_al.transform(X),
        Y)


def test_scaled_procrustes_orthogonalmatrix():
    v = 10
    k = 10
    rnd_matrix = np.random.rand(v, k)
    R, _ = np.linalg.qr(rnd_matrix)
    X = np.random.rand(10, 20)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)
    R_test, _ = scaled_procrustes(X.T, Y.T)
    assert_array_almost_equal(R_test, R)


def test_scaled_procrustes_multiplication():
    X = np.array([[1., 2., 3., 4.],
                  [5., 3., 4., 6.],
                  [7., 8., -5., -2.]])

    X = X - X.mean(axis=1, keepdims=True)

    Y = 2 * X
    Y = Y - Y.mean(axis=1, keepdims=True)

    assert_array_almost_equal(
        scaled_procrustes(X.T, Y.T, scaling=True)[0], np.eye(3))
    assert_array_almost_equal(scaled_procrustes(X.T, Y.T, scaling=True)[1], 2)


def test_scaled_procrustes_basis_orthogonal():
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)

    Y = np.random.rand(3, 4)
    Y = Y - Y.mean(axis=1, keepdims=True)

    R, _ = scaled_procrustes(X.T, Y.T)
    assert_array_almost_equal(R.dot(R.T), np.eye(R.shape[0]))
    assert_array_almost_equal(R.T.dot(R), np.eye(R.shape[0]))


def test_scaled_procrustes_scipy_orthogonal_procrustes():
    X = np.random.rand(4, 4)
    Y = np.random.rand(4, 4)

    R, _ = scaled_procrustes(X, Y)
    R_s, _ = orthogonal_procrustes(Y, X)
    assert_array_almost_equal(R, R_s)


def test_pairwise_scaled_orthogonal_3Drotation():
    R = np.array([[1., 0., 0.], [0., np.cos(1), -np.sin(1)],
                  [0., np.sin(1), np.cos(1)]])
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)

    X = X[:, :, np.newaxis]
    img1_3d = nibabel.Nifti1Image(X, np.eye(4))
    Y = Y[:, :, np.newaxis]
    img2_3d = nibabel.Nifti1Image(Y, np.eye(4))

    mask_img = nibabel.Nifti1Image(np.ones(X.shape, dtype=np.int8), np.eye(4))
    mask_img.shape
    # With mask :
    scaled_orthogonal_with_mask = PairwiseAlignment(
        alignment_method='scaled_orthogonal', mask=mask_img)
    assert_algo_transform_almost_exactly(scaled_orthogonal_with_mask,
                                         img1_3d, img2_3d, mask=mask_img)

    img1_4d = nibabel.Nifti1Image(np.stack([X, X]), np.eye(4))
    img2_4d = nibabel.Nifti1Image(np.stack([Y, Y]), np.eye(4))
    scaled_orthogonal_with_mask = PairwiseAlignment(
        alignment_method='scaled_orthogonal', mask=mask_img)
    assert_algo_transform_almost_exactly(scaled_orthogonal_with_mask,
                                         img1_4d, img2_4d, mask=mask_img)


def test_bagged_pairwise_scaled_orthogonal_3Drotation():
    R = np.array([[1., 0., 0.], [0., np.cos(1), -np.sin(1)],
                  [0., np.sin(1), np.cos(1)]])
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)

    X = X[:, :, np.newaxis]
    Y = Y[:, :, np.newaxis]

    mask_img = nibabel.Nifti1Image(np.ones(X.shape, dtype=np.int8), np.eye(4))
    mask_img.shape

    img1_4d = nibabel.Nifti1Image(np.stack([X, X]), np.eye(4))
    img2_4d = nibabel.Nifti1Image(np.stack([Y, Y]), np.eye(4))
    scaled_orthogonal_with_mask = PairwiseAlignment(
        alignment_method='scaled_orthogonal', n_bags=2, mask=mask_img)

    assert_algo_transform_almost_exactly(scaled_orthogonal_with_mask,
                                         img1_4d, img2_4d, mask=mask_img)


def test_bagged_pairwise_scaled_orthogonal_3Drotation_2jobs():
    R = np.array([[1., 0., 0.], [0., np.cos(1), -np.sin(1)],
                  [0., np.sin(1), np.cos(1)]])
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)

    X = X[:, :, np.newaxis]
    Y = Y[:, :, np.newaxis]

    mask_img = nibabel.Nifti1Image(np.ones(X.shape, dtype=np.int8), np.eye(4))
    mask_img.shape

    img1_4d = nibabel.Nifti1Image(np.stack([X, X]), np.eye(4))
    img2_4d = nibabel.Nifti1Image(np.stack([Y, Y]), np.eye(4))
    scaled_orthogonal_with_mask = PairwiseAlignment(
        alignment_method='scaled_orthogonal', n_bags=2, mask=mask_img, n_jobs=2)

    assert_algo_transform_almost_exactly(scaled_orthogonal_with_mask,
                                         img1_4d, img2_4d, mask=mask_img)


def test_piecewise_scaled_orthogonal_3Drotation():
    R = np.array([[1., 0., 0.], [0., np.cos(1), -np.sin(1)],
                  [0., np.sin(1), np.cos(1)]])
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)

    X = np.stack([X, X])
    Y = np.stack([Y, Y])
    mask_img = nibabel.Nifti1Image(np.ones(X.shape, dtype=np.int8), np.eye(4))

    img1_4d = nibabel.Nifti1Image(np.stack([X, X, X, X, X, X]), np.eye(4))
    img2_4d = nibabel.Nifti1Image(np.stack([Y, Y, Y, Y, Y, Y]), np.eye(4))
    scaled_orthogonal_with_mask = PairwiseAlignment(
        alignment_method='scaled_orthogonal', n_pieces=2, n_bags=20, mask=mask_img)

    assert_algo_transform_almost_exactly(scaled_orthogonal_with_mask,
                                         img1_4d, img2_4d, mask=mask_img)
