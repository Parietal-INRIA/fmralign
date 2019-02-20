import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map
from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts
from fmralign.pairwise_alignment import PairwiseAlignment

# Download images from IBC dataset subjects 1 and 2 (or retrieve them if they were already downloaded)
# Files is the list of paths for each subjects, df is a dataframe with metadata about each of them and mask is an appropriate nifti image to select the data
files, df, mask = fetch_ibc_subjects_contrasts(
    ['sub-01', 'sub-02'])

# Using the mask provided, define a nilearn masker to select relevant data
masker = NiftiMasker(mask_img=mask)
mask
masker.fit()

# Separe the files in training and testing sets for each subjects
im_train_1 = df[df.subject == 'sub-01'][df.acquisition == 'ap'].path.values
im_train_2 = df[df.subject == 'sub-02'][df.acquisition == 'ap'].path.values
im_test_1 = df[df.subject == 'sub-01'][df.acquisition == 'pa'].path.values
im_test_2 = df[df.subject == 'sub-02'][df.acquisition == 'pa'].path.values
# %%
# Define the estimator used to align subjects
alignement_estimator = PairwiseAlignment(
    alignment_method='ridge_cv', n_pieces=150, mask=masker)
# Learn alignment operator for source subject 1 to target subject 2 on training data
alignement_estimator.fit(im_train_1, im_train_2)
# Predict test data for subject 2 from subject 1
im_pred = alignement_estimator.transform(im_test_1)
# Score the prediction of test data without alignment
baseline_score = np.maximum(r2_score(
    masker.transform(im_test_2), masker.transform(im_test_1), multioutput='raw_values'), -1)
# And plot it (save the cut_coords for the next plot)
baseline_display = plot_stat_map(masker.inverse_transform(
    baseline_score), display_mode="z", vmax=0.5)
cut_coords = baseline_display.cut_coords
baseline_display.title("R2 score between raw data")

# Score the prediction made using alignment and plot it as well
aligned_score = np.maximum(r2_score(
    masker.transform(im_test_2), masker.transform(im_pred), multioutput='raw_values'), - 1)
display = plot_stat_map(
    masker.inverse_transform(
        aligned_score), display_mode="z", cut_coords=cut_coords, vmax=0.5)
display.title("R2 score after alignment")

# We can see on the plot that after alignment, the prediction made for one subject data, informed by another subject are greatly improved
