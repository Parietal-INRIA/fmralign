
from nilearn.input_data import NiftiMasker
from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts
from fmralign.pairwise_alignment import PairwiseAlignment

files, df, mask = fetch_ibc_subjects_contrasts(
    ['sub-01', 'sub-02'])

masker = NiftiMasker(mask_img=mask)
mask
masker.fit()


X_train_1 = df[df.subject == 'sub-01'][df.acquisition == 'ap'].path.values
X_train_2 = df[df.subject == 'sub-02'][df.acquisition == 'ap'].path.values
X_test_1 = df[df.subject == 'sub-01'][df.acquisition == 'pa'].path.values
X_test_2 = df[df.subject == 'sub-02'][df.acquisition == 'pa'].path.values
from nilearn.image import load_img
load_img(X_test_2[52])
masker.transform(X_test_2)
X_test_2
parameters = [("identity", 1, 1, "Identity"),
              ("scaled_orthogonal", 1, 1, "Fullbrain Scaled Orthogonal"),
              ("scaled_orthogonal", 150, 1, "Piecewise Scaled Orthogonal"),
              ("scaled_orthogonal", 150, 5, "Bagged Piecewise Scaled Orthogonal")]

for alignment_method, n_pieces, n_bags, method_label in parameters:
    alignement_estimator = PairwiseAlignment(
        alignment_method=alignment_method, n_pieces=n_pieces, n_bags=n_bags, mask=masker)
    alignement_estimator.fit(X_train_1, X_train_2)
    X_pred = alignement_estimator.transform(X_train_1)
