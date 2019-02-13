from nilearn.datasets import fetch_neurovault, neurovault, fetch_neurovault_ids
from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts
from fmralign.pairwise_alignment import PairwiseAlignment

im_1_train, im_2_train(RL), LR
im_1_test, im_2_test

"%s_stacked_contrasts_%s.nii.gz"
masker =

methods_informations = [("identity", 1, 1, "Identity"),
                        ("scaled_orthogonal", 1, 1, "Fullbrain Scaled Orthogonal"),
                        ("scaled_orthogonal", 150, 1,
                         "Piecewise Scaled Orthogonal"),
                        ("scaled_orthogonal", 150, 5, "Bagged Piecewise Scaled Orthogonal")]


for method_alignment, n_pieces, labels in

methods_fullbrain = ["identity", "scaled_orthogonal", "ridge_cv"]
for method in methods_fullbrain:
    estimator = PairwiseAlignment(
        method_alignment=method_alignment, n_pieces=n_pieces)
methods_piecewise = ["permutation", "optimal_transport"]
for method in methods_piecewise:
    estimator = PairwiseAlignment(method_alignment=method, n_pieces=150)
