import nibabel
import numpy as np
from nilearn.maskers import NiftiMasker
from nilearn.surface import InMemoryMesh, PolyMesh, SurfaceImage
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal

from fmralign._utils import _make_parcellation


def zero_mean_coefficient_determination(
    y_true, y_pred, sample_weight=None, multioutput="uniform_average"
):
    """
    Calculate ratio for y_true, y_pred distance to y_true.
    Optimally weights that calculation by provided sample weights.

    Parameters
    ----------
    y_true: (n_samples, n_features) nd array
        Observed y values
    y_pred: (n_samples, n_features) nd array
        Predicted y values
    sample_weight: (n_samples) nd array
        Weighting for each sample.
        Must have matching n_samples as y_true.
    mutlioutput: str
        Must be in ["raw_values", "uniform_average", "variance_weighted"]
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if sample_weight is not None:
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.0

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = (weight * (y_true) ** 2).sum(axis=0, dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (
        numerator[valid_score] / denominator[valid_score]
    )
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0

    if multioutput == "raw_values":
        # return scores individually
        return output_scores
    elif multioutput == "uniform_average":
        # passing None as weights results is uniform mean
        avg_weights = None
    elif multioutput == "variance_weighted":
        avg_weights = (
            weight
            * (y_true - np.average(y_true, axis=0, weights=sample_weight)) ** 2
        ).sum(axis=0, dtype=np.float64)
        # avoid fail on constant y or one-element arrays
        if not np.any(nonzero_denominator):
            if not np.any(nonzero_numerator):
                return 1.0
            else:
                return 0.0
    else:
        avg_weights = multioutput

    return np.average(output_scores, weights=avg_weights)


def assert_class_align_better_than_identity(algo, X, Y):
    """
    Tests that the given algorithm aligns ndarrays better than identity.
    This alignment is measured through r2 score.
    """
    algo.fit(X, Y)
    identity_baseline_score = zero_mean_coefficient_determination(Y, X)
    algo_score = zero_mean_coefficient_determination(Y, algo.transform(X))
    assert algo_score >= identity_baseline_score


def assert_algo_transform_almost_exactly(algo, img1, img2, masker):
    """
    Tests that the given algorithm manages to transform (almost exactly)
    Nifti image img1 into Nifti Image img2.
    """
    algo.fit(img1, img2)
    imtest = algo.transform(img1)
    assert_array_almost_equal(
        masker.transform(img2), masker.transform(imtest), decimal=6
    )


def random_niimg(shape):
    """Produces a random NIfTI image and corresponding mask."""
    rng = default_rng()
    im = nibabel.Nifti1Image(
        rng.random(size=shape, dtype="float32"),
        np.eye(4),
    )
    mask_img = nibabel.Nifti1Image(np.ones(shape[0:3]), np.eye(4))
    return im, mask_img


def sample_parceled_data(n_pieces=1):
    """Create sample data for testing"""
    img, mask_img = random_niimg((8, 7, 6, 20))
    masker = NiftiMasker(mask_img=mask_img)
    data = masker.fit_transform(img)
    labels = _make_parcellation(img, "kmeans", n_pieces, masker)
    return data, masker, labels


def _make_mesh():
    """Create a sample mesh with two parts: left and right, and total of
    9 vertices and 10 faces.

    The left part is a tetrahedron with four vertices and four faces.
    The right part is a pyramid with five vertices and six faces.
    """
    left_coords = np.asarray([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    left_faces = np.asarray([[1, 0, 2], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    right_coords = (
        np.asarray([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
        + 2.0
    )
    right_faces = np.asarray(
        [
            [0, 1, 4],
            [0, 3, 1],
            [1, 3, 2],
            [1, 2, 4],
            [2, 3, 4],
            [0, 4, 3],
        ]
    )
    return PolyMesh(
        left=InMemoryMesh(left_coords, left_faces),
        right=InMemoryMesh(right_coords, right_faces),
    )


def surf_img(n_samples=1):
    """Create a sample surface image using the sample mesh. # noqa: D202
    This will add some random data to the vertices of the mesh.
    The shape of the data will be (n_vertices, n_samples).
    n_samples by default is 1.
    """

    mesh = _make_mesh()
    data = {}
    for key, val in mesh.parts.items():
        data_shape = (val.n_vertices, n_samples)
        data_part = np.random.randn(*data_shape)
        data[key] = data_part
    return SurfaceImage(mesh, data)


def sample_subjects_data(n_subjects=3):
    """Sample data in one parcel for n_subjects"""
    subjects_data = [np.random.rand(10, 20) for _ in range(n_subjects)]
    return subjects_data
