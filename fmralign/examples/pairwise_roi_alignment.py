import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map, plot_roi
from nilearn.image import resample_to_img, load_img, new_img_like
from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts
from fmralign.alignment_methods import ScaledOrthogonalAlignment, RidgeAlignment, DiagonalAlignment, OptimalTransportAlignment
from fmralign.pairwise_alignment import PairwiseAlignment
%matplotlib inline
# Download images from IBC dataset subjects 1 and 2 (or retrieve them if they were already downloaded)
# Files is the list of paths for each subjects, df is a dataframe with metadata about each of them and mask is an appropriate nifti image to select the data

files, df, mask = fetch_ibc_subjects_contrasts(
    ['sub-01', 'sub-02'])

# We also fetch Yeo atlas from nilearn and extract from it a mask for the visual cortex
# Fetch and plot the original Atlas
atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
atlas_yeo = atlas_yeo_2011.thick_7
atlas = load_img(atlas_yeo)
plot_roi(atlas_yeo, title='Original Yeo atlas',
         cut_coords=(8, -80, 9), colorbar=True, cmap='Paired')
# Select visual cortex, create a mask and resample it to the resolution of the images
mask_visual = new_img_like(atlas, atlas.get_data() == 1)
resampled_mask_visual = resample_to_img(
    mask_visual, mask, interpolation="nearest")
# Plot the mask we will use
plot_roi(resampled_mask_visual, title='Visual mask extracted from atlas',
         cut_coords=(8, -80, 9), colorbar=True, cmap='Paired')
# Create a masker using this mask to extract data from the selected region
roi_masker = NiftiMasker(mask_img=resampled_mask_visual)
roi_masker.fit()

# Separe the files in training and testing sets for each subjects
im_train_1 = df[df.subject == 'sub-01'][df.acquisition == 'ap'].path.values
im_train_2 = df[df.subject == 'sub-02'][df.acquisition == 'ap'].path.values
im_test_1 = df[df.subject == 'sub-01'][df.acquisition == 'pa'].path.values
im_test_2 = df[df.subject == 'sub-02'][df.acquisition == 'pa'].path.values

# Define the estimator used to align subjects data
alignment_class = OptimalTransportAlignment()
# Mask the data and learn alignment from source subject 1 to target subject 2 on training data
alignment_class.fit(roi_masker.transform(im_train_1),
                    roi_masker.transform(im_train_2))
# Predict test data for subject 2 from subject 1
predicted_data = alignment_class.transform(roi_masker.transform(im_test_1))
# Mask the real test data for subject 2 to get a ground truth vector
ground_truth = roi_masker.transform(im_test_2)

# Score the prediction of test data without alignment
baseline_score = np.maximum(r2_score(
    ground_truth, roi_masker.transform(im_test_1), multioutput='raw_values'), -1)
# And plot it (save the cut_coords for the next plot)
baseline_display = plot_stat_map(roi_masker.inverse_transform(
    baseline_score), display_mode="z", vmax=0.5)
cut_coords = baseline_display.cut_coords
baseline_display.title("R2 score between raw data")

# Score the prediction made using alignment and plot it as well
aligned_score = np.maximum(r2_score(
    ground_truth, predicted_data, multioutput='raw_values'), -1)

display = plot_stat_map(
    roi_masker.inverse_transform(
        aligned_score), display_mode="z", cut_coords=cut_coords, vmax=0.5)
display.title("R2 score after alignment")

# We can see on the plot that after alignment, the prediction made for one subject data, informed by another subject are greatly improved.

# Instead of masking the data and applying alignment separately. We could also be have done the same directly using PairwiseAlignment() with the visual mask, on nifti images

alignment_estimator = PairwiseAlignment(
    alignment_method='optimal_transport', n_pieces=150, mask=roi_masker)
alignment_estimator.fit(im_train_1, im_train_2)
directly_predicted_img = alignment_estimator.transform(im_test_1)
directly_aligned_score = np.maximum(r2_score(
    ground_truth, roi_masker.transform(directly_predicted_img), multioutput='raw_values'), -1)
display = plot_stat_map(
    roi_masker.inverse_transform(
        directly_aligned_score), display_mode="z", cut_coords=cut_coords, vmax=0.5)
display.title("R2 score after alignment (using PairwiseAlignment)")
