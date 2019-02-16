
from nilearn.input_data import NiftiMasker
from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts
from fmralign.pairwise_alignment import PairwiseAlignment

files, df, mask = fetch_ibc_subjects_contrasts(
    ['sub-01', 'sub-02'])

masker = NiftiMasker(mask_img=mask)
masker.fit()

parameters = [("identity", 1, 1, "Identity"),
              ("scaled_orthogonal", 1, 1, "Fullbrain Scaled Orthogonal"),
              ("scaled_orthogonal", 150, 1,
               "Piecewise Scaled Orthogonal"),
              ("scaled_orthogonal", 150, 5, "Bagged Piecewise Scaled Orthogonal")]

for method_alignment, n_pieces, labels in parameters:

methods_fullbrain = ["identity", "scaled_orthogonal", "ridge_cv"]
for method in methods_fullbrain:
    estimator = PairwiseAlignment(
        method_alignment=method_alignment, n_pieces=n_pieces)
methods_piecewise = ["permutation", "optimal_transport"]
for method in methods_piecewise:
    estimator = PairwiseAlignment(method_alignment=method, n_pieces=150)
