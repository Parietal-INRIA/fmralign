import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from fmralign.alignment_methods import IndividualizedNeuralTuning as INT
from fmralign.generate_data import generate_dummy_signal, generate_dummy_searchlights
from fmralign.hyperalignment.correlation import (
    tuning_correlation,
    stimulus_correlation,
    compute_pearson_corr,
    matrix_MDS,
)


#############################################################################
# INT
#############################################################################


file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


n_s = 10
n_t = 200
n_v = 200
S_std = 1
T_std = 1
SNR = 100
latent_dim = 6  # if None, latent_dim = n_t
decomposition_method = None  # if None, SVD is used


#############################################################################
# GENERATE DUMMY SIGNAL

(
    data_run_1,
    data_run_2,
    stimulus_run_1,
    stimulus_run_2,
    data_tuning,
) = generate_dummy_signal(
    n_s=n_s,
    n_t=n_t,
    n_v=n_v,
    S_std=S_std,
    T_std=T_std,
    latent_dim=latent_dim,
    SNR=SNR,
    seed=0,
)

searchlights, dists = generate_dummy_searchlights(n_searchlights=12, n_v=n_v, radius=5)

#############################################################################
# Test INT on the two parts of the data (ie different runs of the experiment)
int1 = INT(n_components=latent_dim, decomp_method=decomposition_method)
int2 = INT(n_components=latent_dim, decomp_method=decomposition_method)
int_first_part = int1.fit(
    data_run_1,
    searchlights=searchlights,
    dists=dists,
)  # S is provided if we cheat and know the ground truth
int_second_part = int2.fit(data_run_2, searchlights=searchlights, dists=dists)

print("INT 1 denoised signal", int1.denoised_signal.shape)

data_pred = int1.transform(data_run_2)
# save individual components

tuning_pred_run_1 = int1.tuning_data
tuning_pred_run_1 = np.array(tuning_pred_run_1)
tuning_pred_run_2 = int2.tuning_data
tuning_pred_run_2 = np.array(tuning_pred_run_2)

stimulus_pred_run_1 = int1.shared_response
stimulus_pred_run_2 = int2.shared_response


#############################################################################
# Plot
plt.rc("font", size=6)
fig, ax = plt.subplots(3, 3, figsize=(20, 10))


# Tunning matrices
correlation_tuning = tuning_correlation(tuning_pred_run_1, tuning_pred_run_2)
ax[0, 0].imshow(correlation_tuning)
ax[0, 0].set_title("Run 1 vs Run 2")
ax[0, 0].set_xlabel("Subjects, Run 1")
ax[0, 0].set_ylabel("Subjects, Run 2")
fig.colorbar(ax[0, 0].imshow(correlation_tuning), ax=ax[0, 0])

random_colors = np.random.rand(n_s, 3)
# MDS of predicted images
print("X_run_2_pred.shape", data_pred.shape)
print("X_run_2.shape", data_run_2.shape)
corr_tunning = compute_pearson_corr(data_pred, data_run_2)
data_pred_reduced, data_test_reduced = matrix_MDS(
    data_pred, data_run_2, n_components=2, dissimilarity=1 - corr_tunning
)

ax[0, 1].scatter(
    data_pred_reduced[:, 0],
    data_pred_reduced[:, 1],
    label="Run 1",
    c=random_colors,
)
ax[0, 1].scatter(
    data_test_reduced[:, 0],
    data_test_reduced[:, 1],
    label="Run 2",
    c=random_colors,
)
ax[0, 1].set_title("MDS of predicted images, dim=2")

# MDS of tunning matrices
corr_tunning = compute_pearson_corr(tuning_pred_run_1, tuning_pred_run_2)
T_first_part_transformed, T_second_part_transformed = matrix_MDS(
    tuning_pred_run_1, tuning_pred_run_2, n_components=2, dissimilarity=1 - corr_tunning
)


ax[0, 2].scatter(
    T_first_part_transformed[:, 0],
    T_first_part_transformed[:, 1],
    label="Run 1",
    c=random_colors,
)
ax[0, 2].scatter(
    T_second_part_transformed[:, 0],
    T_second_part_transformed[:, 1],
    label="Run 2",
    c=random_colors,
)
ax[0, 2].set_title("MDS of tunning matrices, dim=2")

# Stimulus matrix

correlation_stimulus_true_est_first_part = stimulus_correlation(
    stimulus_pred_run_1.T, stimulus_run_1.T
)
ax[1, 0].imshow(correlation_stimulus_true_est_first_part)
ax[1, 0].set_title("Stimumus Run 1 Estimated vs ground truth")
ax[1, 0].set_xlabel("Latent components, Run 1")
ax[1, 0].set_ylabel("Latent components, ground truth")
fig.colorbar(ax[1, 0].imshow(correlation_stimulus_true_est_first_part), ax=ax[1, 0])

correlation_stimulus_true_est_second_part = stimulus_correlation(
    stimulus_pred_run_2.T, stimulus_run_2.T
)
ax[1, 1].imshow(correlation_stimulus_true_est_second_part)
ax[1, 1].set_title("Stimulus Run 2 Estimated vs ground truth")
ax[1, 1].set_xlabel("Latent components, Run 2")
ax[1, 1].set_ylabel("Latent components, ground truth")
fig.colorbar(ax[1, 1].imshow(correlation_stimulus_true_est_second_part), ax=ax[1, 1])


# Reconstruction
correlation_reconstruction_first_second = tuning_correlation(data_pred, data_run_2)
ax[1, 2].imshow(correlation_reconstruction_first_second)
ax[1, 2].set_title("Reconstruction")
ax[1, 2].set_xlabel("Subjects, Run 2")
ax[1, 2].set_ylabel("Subjects, Run 1")
fig.colorbar(ax[1, 2].imshow(correlation_reconstruction_first_second), ax=ax[1, 2])


plt.rc("font", size=10)
# Define small font for titles
fig.suptitle(
    f"Tunning matrices for the two parts of the data\n ns={n_s}, nt={n_t}, nv={n_v}, S_std={S_std}, T_std={T_std}, SNR={SNR}, latent space dim={latent_dim}"
)
plt.tight_layout()

plt.show()
