# -*- coding: utf-8 -*-
"""
Template-based prediction.
==========================

In this tutorial, we show how to improve inter-subject similarity using a template
computed across multiple source subjects. For this purpose, we create a template
using Procrustes alignment (hyperalignment) to which we align the target subject,
using shared information. We then compare the voxelwise similarity between the
target subject and the template to the similarity between the target subject and
the anatomical Euclidean average of the source subjects.

We mostly rely on Python common packages and on nilearn to handle
functional data in a clean fashion.


To run this example, you must launch IPython via ``ipython
--matplotlib`` in a terminal, or use ``jupyter-notebook``.
"""

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
    ["sub-01", "sub-02", "sub-04", "sub-05", "sub-06", "sub-07"]
)

###############################################################################
# Define a masker
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
for i in range(5):
    template_train.append(concat_imgs(imgs[i]))

# sub-07 (that is 5th in the list) will be our left-out subject.
# We make a single 4D Niimg from our list of 3D filenames.

left_out_subject = concat_imgs(imgs[5])

###############################################################################
# Compute a baseline (average of subjects)
# ----------------------------------------
# We create an image with as many contrasts as any subject representing for
# each contrast the average of all train subjects maps.

import numpy as np

masked_imgs = [masker.transform(img) for img in template_train]
average_img = np.mean(masked_imgs, axis=0)
average_subject = masker.inverse_transform(average_img)

###############################################################################
# Create a template from the training subjects.
# ---------------------------------------------
# We define an estimator using the class TemplateAlignment:
#   * We align the whole brain through 'multiple' local alignments.
#   * These alignments are calculated on a parcellation of the brain in 50 pieces,
#     this parcellation creates group of functionnally similar voxels.
#   * The template is created iteratively, aligning all subjects data into a
#     common space, from which the template is inferred and aligning again to this
#     new template space.
#

from fmralign.template_alignment import TemplateAlignment

# We use Procrustes/scaled orthogonal alignment method
template_estim = TemplateAlignment(
    n_pieces=50,
    alignment_method="scaled_orthogonal",
    masker=masker,
)
template_estim.fit(template_train)
procrustes_template = template_estim.template

###############################################################################
# Predict new data for left-out subject
# -------------------------------------
# We predict the contrasts of the left-out subject using the template we just
# created. We use the transform method of the estimator. This method takes the
# left-out subject as input, computes a pairwise alignment with the template
# and returns the aligned data.

predictions_from_template = template_estim.transform(left_out_subject)

###############################################################################
# Score the baseline and the prediction
# -------------------------------------
# We use a utility scoring function to measure the voxelwise correlation
# between the images. That is, for each voxel, we measure the correlation between
# its profile of activation without and with alignment, to see if template-based
# alignment was able to improve inter-subject similarity.

from fmralign.metrics import score_voxelwise

average_score = masker.inverse_transform(
    score_voxelwise(left_out_subject, average_subject, masker, loss="corr")
)
template_score = masker.inverse_transform(
    score_voxelwise(
        predictions_from_template, procrustes_template, masker, loss="corr"
    )
)

###############################################################################
# Plotting the measures
# ---------------------
# Finally we plot both scores
#

from nilearn import plotting

baseline_display = plotting.plot_stat_map(
    average_score, display_mode="z", vmax=1, cut_coords=[-15, -5]
)
baseline_display.title("Left-out subject correlation with group average")
display = plotting.plot_stat_map(
    template_score, display_mode="z", cut_coords=[-15, -5], vmax=1
)
display.title("Aligned subject correlation with Procrustes template")

###############################################################################
# We observe that creating a template and aligning a new subject to it yields
# better inter-subject similarity than regular euclidean averaging.
