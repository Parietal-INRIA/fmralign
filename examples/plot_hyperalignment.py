# -*- coding: utf-8 -*-

"""
Hyperalignment-base prediction using the IndividualNeuralTuning Model.
See article : https://doi.org/10.1162/imag_a_00032

==========================

In this tutorial, we show how to better predict new contrasts for a target
subject using many source subjects corresponding contrasts. For this purpose,
we create a template to which we align the target subject, using shared information.
We then predict new images for the target and compare them to a baseline.

We mostly rely on Python common packages and on nilearn to handle
functional data in a clean fashion.


To run this example, you must launch IPython via ``ipython
--matplotlib`` in a terminal, or use ``jupyter-notebook``.

.. contents:: **Contents**
    :local:
    :depth: 1

"""
# %%
import warnings

warnings.filterwarnings("ignore")
###############################################################################
# Retrieve the data
# -----------------
# In this example we use the IBC dataset, which includes a large number of
# different contrasts maps for 12 subjects.
# We download the images for subjects sub-01, sub-02, sub-04, sub-05, sub-06
# and sub-07 (or retrieve them if they were already downloaded).
# imgs is the list of paths to available statistical images for each subjects.
# df is a dataframe with metadata about each of them.
# mask is a binary image used to extract grey matter regions.
#

from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts

imgs, df, mask_img = fetch_ibc_subjects_contrasts(
    [
        "sub-01",
        "sub-02",
        "sub-04",
        "sub-05",
        "sub-06",
        "sub-07",
    ],
)

SEARCHLIGHT = False

###############################################################################
# Definine a masker
# -----------------
# We define a nilearn masker that will be used to handle relevant data.
#   For more information, visit :
#   'http://nilearn.github.io/manipulating_images/masker_objects.html'
#

from nilearn.maskers import NiftiMasker

masker = NiftiMasker(mask_img=mask_img).fit()

###############################################################################
# Prepare the data
# ----------------
# For each subject, we will use two series of contrasts acquired during
# two independent sessions with a different phase encoding:
# Antero-posterior(AP) or Postero-anterior(PA).
#


# To infer a template for subjects sub-01 to sub-06 for both AP and PA data,
# we make a list of 4D niimgs from our list of list of files containing 3D images

from nilearn.image import concat_imgs

template_train = []
for i in range(6):
    template_train.append(concat_imgs(imgs[i]))
target_train = df[df.subject == "sub-07"][df.acquisition == "ap"].path.values

# For subject sub-07, we split it in two folds:
#   - target train: sub-07 AP contrasts, used to learn alignment to template
#   - target test: sub-07 PA contrasts, used as a ground truth to score predictions
# We make a single 4D Niimg from our list of 3D filenames

target_train = concat_imgs(target_train)
target_train_data = masker.transform(target_train)
target_test = df[df.subject == "sub-07"][df.acquisition == "pa"].path.values


###############################################################################
# Compute a baseline (average of subjects)
# ----------------------------------------
# We create an image with as many contrasts as any subject representing for
# each contrast the average of all train subjects maps.
#

import numpy as np

masked_imgs = [masker.transform(img) for img in template_train]
average_img = np.mean(masked_imgs[:-1], axis=0)
average_subject = masker.inverse_transform(average_img)

###############################################################################
# Create a template from the training subjects.
# ---------------------------------------------
# We define an estimator using the class TemplateAlignment:
#   * We align the whole brain through 'multiple' local alignments.
#   * These alignments are calculated on a parcellation of the brain in 150 pieces,
#     this parcellation creates group of functionnally similar voxels.
#   * The template is created iteratively, aligning all subjects data into a
#     common space, from which the template is inferred and aligning again to this
#     new template space.
#

from fmralign.hyperalignment.regions import compute_parcels, compute_searchlights

###############################################################################
# Predict new data for left-out subject
# -------------------------------------
# We use target_train data to fit the transform, indicating it corresponds to
# the contrasts indexed by train_index and predict from this learnt alignment
# contrasts corresponding to template test_index numbers.
# For each train subject and for the template, the AP contrasts are sorted from
# 0, to 53, and then the PA contrasts from 53 to 106.
#

train_index = range(53)
test_index = range(53, 106)

train_data = np.array(masked_imgs)[:, train_index, :]
test_data = np.array(masked_imgs)[:, test_index, :][:-1]

if not SEARCHLIGHT:
    parcels = compute_parcels(
        niimg=template_train[0], mask=masker, n_parcels=1000, n_jobs=5
    )

else:
    _, searchlights, dists = compute_searchlights(
        niimg=template_train[0], mask_img=masker.mask_img, radius=5, n_jobs=5
    )

# %%

from fmralign.hyperalignment.regions import template, piece_ridge, searchlight_weights

if not SEARCHLIGHT:
    sl_template = template(
        X=train_data, regions=parcels, n_jobs=5, template_kind="procrustes"
    )
    train_tuning = piece_ridge(X=sl_template, Y=train_data[-1], regions=parcels)
    test_template = template(X=test_data, regions=parcels, n_jobs=5)

    train_tuning = piece_ridge(
        X=sl_template, Y=train_data[-1], regions=parcels, return_betas=True
    )
else:
    weights = searchlight_weights(searchlights=searchlights, dists=dists, radius=20)
    sl_template = template(
        X=train_data, regions=searchlights, n_jobs=5, weights=weights
    )
    train_tuning = piece_ridge(
        X=sl_template,
        Y=train_data[-1],
        regions=searchlights,
        weights=weights,
        alpha=10,
        return_betas=True,
    )
    test_template = template(
        X=test_data, regions=searchlights, n_jobs=5, weights=weights
    )

# %%

target_pred = test_template @ train_tuning

target_pred = masker.inverse_transform(target_pred)

# %%
from fmralign.metrics import score_voxelwise

# Now we use this scoring function to compare the correlation of predictions
# made from group average and from template with the real PA contrasts of sub-07


template_score = masker.inverse_transform(
    C_temp := score_voxelwise(target_test, target_pred, masker=masker, loss="corr")
)

print("============Global correlation============")
print(f"Template avg : {np.mean(C_temp)}")
print(f"Template std : {np.std(C_temp)}")

###############################################################################
# Plotting the measures
# ---------------------
# Finally we plot both scores
#

# %%
from nilearn import plotting

display = plotting.plot_stat_map(
    template_score, display_mode="z", cut_coords=[-15, -5], vmax=1
)
# display.title("Hyperalignment-based prediction correlation wt ground truth")

###############################################################################
# We observe that creating a template and aligning a new subject to it yields
# a prediction that is better correlated with the ground truth than just using
# the average activations of subjects.
#


# %%
from nilearn.image import load_img
from nilearn.maskers import NiftiMasker

moviewatching_mask_img = load_img("/Users/df/nilearn_data/movie_watching.nii.gz")
moviewatching_mask_img = masker.transform(moviewatching_mask_img)
moviewatching_mask_img -= np.mean(moviewatching_mask_img, axis=0)
moviewatching_mask_img /= np.std(moviewatching_mask_img, axis=0)
moviewatching_mask_img = moviewatching_mask_img > 0
moviewatching_mask_img = masker.inverse_transform(moviewatching_mask_img)
moviewatching_masker = NiftiMasker(mask_img=moviewatching_mask_img).fit()


target_pred = moviewatching_masker.transform(target_pred)
target_test = moviewatching_masker.transform(target_test)

target_pred = target_pred - np.mean(target_pred, axis=0)

C_temp_localized = score_voxelwise(
    target_test, target_pred, masker=moviewatching_masker, loss="corr"
)

template_localized_score = moviewatching_masker.inverse_transform(C_temp)

print("============Global correlation============")
print(f"Template avg : {np.mean(C_temp)}")
print(f"Template std : {np.std(C_temp)}")

localized_display = plotting.plot_stat_map(
    template_score, display_mode="z", cut_coords=[-15, -5], vmax=1
)


plotting.show()
