import numpy as np
import pytest
from nibabel import Nifti1Image
from nilearn.image import concat_imgs, math_img
from nilearn.maskers import NiftiMasker
from nilearn.surface import SurfaceImage
from numpy.testing import assert_array_almost_equal

from fmralign._utils import ParceledData
from fmralign.preprocessing import ParcellationMasker
from fmralign.template_alignment import (
    TemplateAlignment,
    _align_images_to_template,
    _fit_local_template,
    _index_by_parcel,
    _reconstruct_template,
    _rescaled_euclidean_mean,
)
from fmralign.tests.utils import (
    random_niimg,
    sample_parceled_data,
    sample_subjects_data,
    surf_img,
    zero_mean_coefficient_determination,
)


@pytest.mark.parametrize("scale_average", [True, False])
def test_rescaled_euclidean_mean(scale_average):
    subjects_data = sample_subjects_data()
    average_data = _rescaled_euclidean_mean(subjects_data)
    assert average_data.shape == subjects_data[0].shape
    assert average_data.dtype == subjects_data[0].dtype

    if scale_average is False:
        assert np.allclose(average_data, np.mean(subjects_data, axis=0))


def test_reconstruct_template():
    n_subjects = 3
    n_iter = 3
    n_pieces = 2
    imgs = [random_niimg((8, 7, 6, 20))[0]] * n_subjects
    parcel_masker = ParcellationMasker(n_pieces=n_pieces)
    subjects_parcels = parcel_masker.fit_transform(imgs)
    parcels_subjects = _index_by_parcel(subjects_parcels)
    masker = parcel_masker.masker
    labels = parcel_masker.labels

    fit = [
        _fit_local_template(parcel_i, n_iter=n_iter)
        for parcel_i in parcels_subjects
    ]
    template, template_history = _reconstruct_template(fit, labels, masker)

    assert template.shape == imgs[0].shape
    assert len(template_history) == n_iter - 2
    for template_i in template_history:
        assert template_i.shape == imgs[0].shape


def test_align_images_to_template():
    subjects_data = sample_subjects_data()
    template = _rescaled_euclidean_mean(subjects_data)
    aligned_data, subjects_estimators = _align_images_to_template(
        subjects_data,
        template,
        alignment_method="identity",
    )
    assert len(aligned_data) == len(subjects_data)
    assert len(subjects_estimators) == len(subjects_data)
    assert aligned_data[0].shape == subjects_data[0].shape


def test_fit_local_template():
    n_subjects = 3
    n_iter = 3
    subjects_data = sample_subjects_data(n_subjects=n_subjects)
    fit = _fit_local_template(
        subjects_data,
        n_iter=n_iter,
        alignment_method="identity",
        scale_template=False,
    )
    template_data = fit["template_data"]
    template_history = fit["template_history"]
    estimators = fit["estimators"]

    assert template_data.shape == subjects_data[0].shape
    assert len(template_history) == n_iter - 2
    assert len(estimators) == n_subjects


def test_index_by_parcel():
    n_subjects = 3
    n_pieces = 2
    subjects_parcels = [
        ParceledData(*sample_parceled_data(n_pieces))
        for _ in range(n_subjects)
    ]
    parcels_subjects = _index_by_parcel(subjects_parcels)
    assert len(parcels_subjects) == n_pieces
    assert len(parcels_subjects[0]) == n_subjects
    assert parcels_subjects[0][0].shape == subjects_parcels[0][0].shape


def test_template_identity():
    n = 10
    im, mask_img = random_niimg((6, 5, 3))

    sub_1 = concat_imgs(n * [im])
    sub_2 = math_img("2 * img", img=sub_1)
    sub_3 = math_img("3 * img", img=sub_1)

    ref_template = sub_2
    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()

    subs = [sub_1, sub_2, sub_3]

    # test different fit() accept list of list of 3D Niimgs as input.
    algo = TemplateAlignment(alignment_method="identity", masker=masker)
    algo.fit([concat_imgs(n * [im])] * 3)
    # test template
    assert_array_almost_equal(sub_1.get_fdata(), algo.template.get_fdata())

    # test fit() transform() with 4D Niimgs input for several params set
    args_list = [
        {"alignment_method": "identity", "masker": masker},
        {"alignment_method": "identity", "masker": masker, "n_jobs": 2},
        {"alignment_method": "identity", "n_pieces": 3, "masker": masker},
    ]

    for args in args_list:
        algo = TemplateAlignment(**args)
        algo.fit(subs)
        # test template
        assert_array_almost_equal(
            ref_template.get_fdata(),
            algo.template.get_fdata(),
        )
        predicted_imgs = algo.transform(ref_template)
        assert_array_almost_equal(
            predicted_imgs.get_fdata(),
            ref_template.get_fdata(),
        )
        predicted_imgs = algo.transform(ref_template, subject_index=1)
        assert_array_almost_equal(
            predicted_imgs.get_fdata(),
            ref_template.get_fdata(),
        )


def test_template_diagonal():
    n = 10
    im, mask_img = random_niimg((6, 5, 3))

    sub_1 = concat_imgs(n * [im])
    sub_2 = math_img("2 * img", img=sub_1)
    sub_3 = math_img("3 * img", img=sub_1)

    ref_template = sub_2
    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()

    subs = [sub_1, sub_2, sub_3]

    # Test without subject_index
    algo = TemplateAlignment(alignment_method="diagonal", masker=masker)
    algo.fit(subs)
    predicted_imgs = algo.transform(sub_1, subject_index=None)
    assert_array_almost_equal(
        ref_template.get_fdata(),
        predicted_imgs.get_fdata(),
    )

    # Test with subject_index
    for i, sub in enumerate(subs):
        predicted_imgs = algo.transform(sub, subject_index=i)
        assert_array_almost_equal(
            predicted_imgs.get_fdata(),
            ref_template.get_fdata(),
        )


def test_template_closer_to_target():
    n_samples = 6

    subject_1, mask_img = random_niimg((6, 5, 3, n_samples))
    subject_2, _ = random_niimg((6, 5, 3, n_samples))

    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()

    # Calculate metric between each subject and the average

    sub_1 = masker.transform(subject_1)
    sub_2 = masker.transform(subject_2)
    subs = [subject_1, subject_2]
    avg_data = np.mean([sub_1, sub_2], axis=0)
    mean_distance_1 = zero_mean_coefficient_determination(sub_1, avg_data)
    mean_distance_2 = zero_mean_coefficient_determination(sub_2, avg_data)

    for alignment_method in [
        "ridge_cv",
        "scaled_orthogonal",
        "optimal_transport",
        "diagonal",
    ]:
        algo = TemplateAlignment(
            alignment_method=alignment_method,
            n_pieces=3,
            masker=masker,
        )
        # Learn template
        algo.fit(subs)
        # Assess template is closer to mean than both images
        template_data = masker.transform(algo.template)
        template_mean_distance = zero_mean_coefficient_determination(
            avg_data,
            template_data,
        )
        assert template_mean_distance >= mean_distance_1
        assert (
            template_mean_distance >= mean_distance_2 - 1.0e-2
        )  # for robustness


def test_parcellation_retrieval():
    """Test that TemplateAlignment returns both the\n
    labels and the parcellation image
    """
    n_pieces = 3
    imgs = [random_niimg((8, 7, 6))[0]] * 3
    alignment = TemplateAlignment(n_pieces=n_pieces)
    alignment.fit(imgs)

    labels, parcellation_image = alignment.get_parcellation()
    assert isinstance(labels, np.ndarray)
    assert len(np.unique(labels)) == n_pieces
    assert isinstance(parcellation_image, Nifti1Image)
    assert parcellation_image.shape == imgs[0].shape[:-1]


def test_parcellation_before_fit():
    """Test that TemplateAlignment raises an error if\n
    the parcellation is retrieved before fitting
    """
    alignment = TemplateAlignment()
    with pytest.raises(
        AttributeError,
        match="Parcellation has not been computed yet",
    ):
        alignment.get_parcellation()


@pytest.mark.parametrize("modality", ["response", "connectivity", "hybrid"])
def test_various_modalities(modality):
    """Test all modalities with Nifti images"""
    n_pieces = 3
    img1, _ = random_niimg((8, 7, 6, 10))
    img2, _ = random_niimg((8, 7, 6, 10))
    img3, _ = random_niimg((8, 7, 6, 10))
    alignment = TemplateAlignment(n_pieces=n_pieces, modality=modality)

    # Test fitting
    alignment.fit([img1, img2, img3])
    assert isinstance(alignment.template, Nifti1Image)
    if modality == "response":
        assert alignment.template.shape == img1.shape
    elif modality == "connectivity":
        assert alignment.template.shape[-1] == n_pieces
    elif modality == "hybrid":
        assert alignment.template.shape[-1] == n_pieces + img1.shape[-1]

    # Test transformation on existing subject
    img_transformed = alignment.transform(img1, subject_index=0)
    assert isinstance(img_transformed, Nifti1Image)
    assert img_transformed.shape == img1.shape

    # Test transformation on new subject
    new_img, _ = random_niimg((8, 7, 6, 10))
    img_transformed = alignment.transform(new_img)
    assert isinstance(img_transformed, Nifti1Image)
    assert img_transformed.shape == img1.shape


@pytest.mark.parametrize("modality", ["response", "connectivity", "hybrid"])
def test_surface_template(modality):
    """Test compatibility with `SurfaceImage`"""
    n_pieces = 3
    img1 = surf_img(20)
    img2 = surf_img(20)
    img3 = surf_img(20)
    alignment = TemplateAlignment(n_pieces=n_pieces, modality=modality)

    # Test fitting
    alignment.fit([img1, img2, img3])
    assert isinstance(alignment.template, SurfaceImage)

    # Test transformation from new subject
    img_transformed = alignment.transform(surf_img(20))
    assert isinstance(img_transformed, SurfaceImage)
    assert img_transformed.shape == surf_img(20).shape

    # Test transformation on real subject
    img_transformed = alignment.transform(img1, subject_index=0)
    assert isinstance(img_transformed, SurfaceImage)
    masker = alignment.masker
    assert np.allclose(
        masker.transform(img_transformed), masker.transform(img1)
    )

    # Test parcellation retrieval
    labels, parcellation_image = alignment.get_parcellation()
    assert isinstance(labels, np.ndarray)
    assert len(np.unique(labels)) == n_pieces
    assert isinstance(parcellation_image, SurfaceImage)
