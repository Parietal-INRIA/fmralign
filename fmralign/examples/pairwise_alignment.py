import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map
from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts
from fmralign.pairwise_alignment import PairwiseAlignment

files, df, mask = fetch_ibc_subjects_contrasts(
    ['sub-01', 'sub-02'])

masker = NiftiMasker(mask_img=mask)
mask
masker.fit()


im_train_1 = df[df.subject == 'sub-01'][df.acquisition == 'ap'].path.values
im_train_2 = df[df.subject == 'sub-02'][df.acquisition == 'ap'].path.values
im_test_1 = df[df.subject == 'sub-01'][df.acquisition == 'pa'].path.values
im_test_2 = df[df.subject == 'sub-02'][df.acquisition == 'pa'].path.values

alignement_estimator = PairwiseAlignment(
    alignment_method="scaled_orthogonal", n_pieces=150, mask=masker)
alignement_estimator.fit(im_train_1, im_train_2)
im_pred = alignement_estimator.transform(im_test_1)
import numpy as np
labels = alignement_estimator.labels_[0]
estimators = alignement_estimator.fit_[0]
X = masker.transform(im_test_1).T
np.shape(X_test_1)
X_transform = np.zeros_like(X)
i = 0
X_transform[labels == i] = estimators[i].transform(X[labels == i].T).T
estimators[i].transform(X[labels == i].T)

X_transform[labels == i] = estimators[i].transform(X[labels == i].T).T


baseline_score = r2_score(
    masker.transform(im_test_2), masker.transform(im_test_1), multioutput='raw_values')
aligned_score = r2_score(
    masker.transform(im_test_2), masker.transform(im_pred), multioutput='raw_values')


baseline_display = plot_stat_map(masker.inverse_transform(
    baseline_score), display_mode="z", vmax=0.5)
cut_coords = baseline_display.cut_coords
baseline_display.title("R2 score between raw data")

display = plot_stat_map(
    masker.inverse_transform(
        aligned_score), display_mode="z", cut_coords=cut_coords, vmax=0.5)
display.title("R2 score after alignment")
