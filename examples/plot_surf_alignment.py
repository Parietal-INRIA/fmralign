# -*- coding: utf-8 -*-
"""
Pairwise surface alignment.
==============================

In this tutorial, we show how to align surface data from two subjects using
a pairwise alignment method. We project the data on the `fsaverage5` surface
and learn a piecewise mapping from one subject to the other using ward clustering.


We mostly rely on python common packages and on nilearn to handle functional
data in a clean fashion.

"""
###############################################################################
# Retrieve the data
# -----------------
# In this example we use the IBC dataset, which include a large number of
# different contrasts maps for 12 subjects. We download the images for
# subjects sub-01 and sub-02 (or retrieve them if they were already downloaded)
# Files is the list of paths for each subjects.
# df is a dataframe with metadata about each of them.

from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts

files, df, _ = fetch_ibc_subjects_contrasts(["sub-01", "sub-02"])


###############################################################################
# Project the data on the fsaverage5 surface
# ------------------------------------------
# We project the data on the fsaverage5 surface, using the fsaverage5
# surface template.

from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.surface import SurfaceImage

fsaverage_meshes = load_fsaverage()


def project_to_surface(img):
    """Util function for loading and projecting volumetric images."""
    surface_image = SurfaceImage.from_volume(
        mesh=fsaverage_meshes["pial"],
        volume_img=img,
    )
    return surface_image


from nilearn.image import concat_imgs

source_train = concat_imgs(
    df[df.subject == "sub-01"][df.acquisition == "ap"].path.values
)
target_train = concat_imgs(
    df[df.subject == "sub-02"][df.acquisition == "ap"].path.values
)

surf_source_train = project_to_surface(source_train)
surf_target_train = project_to_surface(target_train)


###############################################################################
# Fitting the alignment operator
# ------------------------------
# We use the `PairwiseAlignment` class to learn the alignment operator from
# one subject to the other. We select the `scaled_orthogonal` method to compute
# a rigid piecewise alignment mapping and the `ward` clustering method to
# parcellate the cortical surface.

from fmralign.pairwise_alignment import PairwiseAlignment

alignment_estimator = PairwiseAlignment(
    alignment_method="scaled_orthogonal",
    n_pieces=100,
    clustering="ward",
)
# Learn alignment operator from subject 1 to subject 2 on training data
alignment_estimator.fit(surf_source_train, surf_target_train)


###############################################################################
# Plot the computed parcellation
# ------------------------------
# We can retrieve the computed parcellation and plot it on the surface.
from nilearn import plotting

_, clustering_img = alignment_estimator.get_parcellation()


plotting.plot_surf_roi(
    surf_mesh=fsaverage_meshes["pial"],
    roi_map=clustering_img,
    hemi="left",
    view="lateral",
    title="Ward parcellation on the left hemisphere",
)
plotting.show()

###############################################################################
# Projecting the left-out data
# ----------------------------
# Let's now align a left-out audio contrast from sub-01 to sub-02. We project
# the data on the surface and apply the learned alignment operator.

surf_audio_source = project_to_surface(
    df[
        (df.subject == "sub-01")
        & (df.condition == "audio_sentence")
        & (df.acquisition == "pa")
    ].path.values
)

surf_audio_target = project_to_surface(
    df[
        (df.subject == "sub-02")
        & (df.condition == "audio_sentence")
        & (df.acquisition == "pa")
    ].path.values
)

surf_aligned = alignment_estimator.transform(surf_audio_source)

###############################################################################
# Visualizing the alignment in action
# -----------------------------------
# We interpolate between the source and aligned images to visualize the
# alignment process. Notice how the individual idiocyncracies of the source
# subject are progressively removed.

from copy import deepcopy

import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

fsaverage_sulcal = load_fsaverage_data(
    mesh="fsaverage5",
    data_type="sulcal",
    mesh_type="inflated",
)

plotting_params = {
    "bg_map": fsaverage_sulcal,
    "hemi": "left",
    "view": "lateral",
    "colorbar": True,
    "alpha": 0.5,
    "bg_on_data": True,
    "vmax": 3,
    "vmin": -3,
    "cmap": "coolwarm",
}


def interpolate_surf_image(surf_img1, surf_img2, alpha=0.5):
    """Interpolate two surface images."""
    # Create a new surface image with the same mesh as the input
    surf_img_interpolated = deepcopy(surf_img1)
    # Interpolate the data
    for hemi in ["left", "right"]:
        surf_img_interpolated.data.parts[hemi] = (
            surf_img1.data.parts[hemi] * (1 - alpha)
            + surf_img2.data.parts[hemi] * alpha
        )
    return surf_img_interpolated


# Create figure
fig = plt.figure(figsize=(10, 8))


# Define a function to update the figure for each frame
def update(frame):
    plt.clf()

    if frame <= 10:
        # Interpolation frames (0-10)
        alpha = frame / 10
        surf_interpolated = interpolate_surf_image(
            surf_audio_source, surf_aligned, alpha=alpha
        )
        plotting.plot_surf_stat_map(
            surf_mesh=fsaverage_meshes["pial"],
            stat_map=surf_interpolated,
            figure=fig,
            **plotting_params,
        )
        plt.suptitle(
            f"Interpolated audio sentence alpha={alpha:.1f}", fontsize=16
        )
    else:
        # Target image (frame 10)
        plotting.plot_surf_stat_map(
            surf_mesh=fsaverage_meshes["pial"],
            stat_map=surf_audio_target,
            figure=fig,
            **plotting_params,
        )
        plt.suptitle("Target image", fontsize=16)

    return [fig]


# Create the animation
anim = FuncAnimation(fig, update, frames=range(12), interval=300, blit=True)
