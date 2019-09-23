# -*- coding: utf-8 -*-
""" Alignment methods benchmark (pairwise ROI case)
===================================================================

In this tutorial, We compare various methods of alignment on a pairwise alignment \
problem for `Individual Brain Charting < https: // project.inria.fr / IBC / >`_ subjects.\
For each subject, we have a lot of functional informations in the form of several task-based\
contrast per subject. We will just work here on a ROI.

We mostly rely on python common packages and on nilearn to handle functional \
data in a clean fashion.


To run this example, you must launch IPython via ``ipython \
--matplotlib`` in a terminal, or use ``jupyter-notebook``.

.. contents:: **Contents**
    :local:
    :depth: 1

"""

###############################################################################
#  Retrieve the data
# ---------------------
# In this example we use the IBC dataset, which include a large number of \
# different contrasts maps for 12 subjects. \
# We download the images for subjects 1 and 2.
# Files is the list of paths for each subjects.
# df is a dataframe with metadata about each of them.
#

from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts
files, df, mask = fetch_ibc_subjects_contrasts(
    ['sub-01', 'sub-02'])


###############################################################################
# Extract a mask for the visual cortex from Yeo Atlas
# ----------------------------------------------------
# First, we fetch and plot the complete atlas
#

from nilearn import datasets, plotting
from nilearn.image import resample_to_img, load_img, new_img_like
atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
atlas = load_img(atlas_yeo_2011.thick_7)

# Select visual cortex, create a mask and resample it to the right resolution

mask_visual = new_img_like(atlas, atlas.get_data() == 1)
resampled_mask_visual = resample_to_img(
    mask_visual, mask, interpolation="nearest")

# Plot the mask we will use
plotting.plot_roi(resampled_mask_visual, title='Visual regions mask extracted from atlas',
                  cut_coords=(8, -80, 9), colorbar=True, cmap='Paired')

###############################################################################
# Define a masker
# ----------------
# We define a nilearn masker that will be used to handle relevant data. \
#   For more information, visit : \
#   'http://nilearn.github.io/manipulating_images/masker_objects.html'
#

from nilearn.input_data import NiftiMasker
roi_masker = NiftiMasker(mask_img=resampled_mask_visual).fit()


###############################################################################
# Prepare the data
# -------------------
# For each subject, for each task and conditions, our dataset contains two \
# independent acquisitions, similar except for one acquisition parameter, the \
# encoding phase used that was either Antero-Posterior (AP) or Postero-Anterior (PA).
# Although this induces small differences in the final data, we will take \
# advantage of these "duplicates" to create a training and a testing set that \
# contains roughly the same signals but acquired totally independently.
#

# The training fold, used to learn alignment from source subject toward target:
# * source train: AP contrasts for subject one
# * target train: AP contrasts for subject two

source_train = df[df.subject == 'sub-01'][df.acquisition == 'ap'].path.values
target_train = df[df.subject == 'sub-02'][df.acquisition == 'ap'].path.values

# The testing fold:
# * source test: PA contrasts for subject one, used to predict \
#   the corresponding contrasts of subject two
# * target test: PA contrasts for subject two, used as a ground truth \
#   to score our predictions

source_test = df[df.subject == 'sub-01'][df.acquisition == 'pa'].path.values
target_test = df[df.subject == 'sub-02'][df.acquisition == 'pa'].path.values

###############################################################################
# Choose the number of regions for local alignment
# ---------------------------------------------------------------------------
# First, as we will proceed to local alignment we choose a suitable number of regions \
# so that each of them is approximately 200 voxels wide. Then our estimator will first \
# make a functional clustering of voxels based on train data to divide them into meaningful regions.
import numpy as np
n_voxels = roi_masker.mask_img_.get_data().sum()
print(f"The chosen region of interest contains {n_voxels} voxels")
n_pieces = np.round(n_voxels / 200)
print(f"We will cluster them in {n_pieces} regions")

###############################################################################
# Define the estimators, fit them and do a prediction
# ---------------------------------------------------------------------------
# On each region, we search for a transformation R that is either :
#   *  orthogonal, i.e. R orthogonal, scaling sc s.t. ||sc RX - Y ||^2 is minimized
#   *  a ridge regression : ||XR - Y||^2 + alpha *||R||^2` with a L2 penalization
#       on the norm of R.
#   *  the optimal transport plan, which yields the minimal transport cost while \
#       respecting the mass conservation constraints. Calculated with entropic regularization.
#   *  we also include identity (no alignment) as a baseline.
# Then for each method we define the estimator fit it, predict the new image and plot
# its correlation with the real signal.


from fmralign.pairwise_alignment import PairwiseAlignment
from fmralign._utils import voxelwise_correlation
#
methods = ['identity', 'scaled_orthogonal', 'ridge_cv', 'optimal_transport']

for method in methods:
    alignment_estimator = PairwiseAlignment(
        alignment_method=method, n_pieces=n_pieces, mask=roi_masker)
    alignment_estimator.fit(source_train, target_train)
    target_pred = alignment_estimator.transform(source_test)
    aligned_score = voxelwise_correlation(target_test, target_pred, roi_masker)
    display = plotting.plot_stat_map(aligned_score, display_mode="z",
                                     cut_coords=[-15, -5], vmax=1, title=f"Correlation of prediction after {method} alignment")

#############################################################################
# We can observe that all alignment methods perform better than identity(no alignment). \
# Ridge and Optimal Transport perform better than Scaled Orthogonal alignment. \
# Usually Ridge yields best scores for this kind of metrics but for real world \
# problem we suspect it destroys signal specificity, and yiels very smooth predictions. \
# Our recommandation is to use scaled orthogonal for quick alignments and \
# optimal transport for best alignment.
