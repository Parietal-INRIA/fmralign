from fmralign.alignment_methods import IndividualizedNeuralTuning as INT
from fmralign.generate_data import generate_dummy_signal, generate_dummy_searchlights
import numpy as np


def test_int_fit_predict():
    """Test if the outputs and arguments of the INT are the correct format"""
    # Create random data
    X_train, X_test, S_true_first_part, S_true_second_part, Ts = generate_dummy_signal(
        n_s=5,
        n_t=50,
        n_v=300,
        S_std=1,
        T_std=1,
        latent_dim=6,
        SNR=100,
        generative_method="custom",
        seed=0,
    )
    from fmralign.hyperalignment.correlation import (
        tuning_correlation,
        stimulus_correlation,
    )

    # Testing without searchlights
    searchlights = [np.arange(300)]
    dists = [np.ones((300,))]

    # Test INT on the two parts of the data (ie different runs of the experiment)
    int1 = INT(n_components=6)
    int2 = INT(n_components=6)
    int1.fit(
        X_train=X_train, searchlights=searchlights, dists=dists
    )  # S is provided if we cheat and know the ground truth
    int2.fit(X_test, searchlights=searchlights, dists=dists)

    X_pred = int1.transform(X_test)
    # save individual components

    tuning_data_run_1 = int1.tuning_data
    tuning_data_run_2 = int2.tuning_data
    tuning_data_run_1 = np.array(tuning_data_run_1)
    tuning_data_run_2 = np.array(tuning_data_run_2)

    stimulus_run_1 = int1.shared_response
    S_estimated_second_part = int2.shared_response

    corr1 = tuning_correlation(tuning_data_run_1, tuning_data_run_2)
    corr2 = stimulus_correlation(stimulus_run_1.T, S_true_first_part.T)
    corr3 = stimulus_correlation(S_estimated_second_part.T, S_true_second_part.T)
    corr4 = tuning_correlation(X_pred, X_test)

    # Check that predicted components have the same shape as original data

    # Check that the correlation between the two parts of the data is high
    corr1_out = corr1 - np.diag(corr1)
    corr2_out = corr2 - np.diag(corr2)
    corr3_out = corr3 - np.diag(corr3)
    corr4_out = corr4 - np.diag(corr4)
    assert 3 * np.mean(corr1_out) < np.mean(np.diag(corr1))
    assert 3 * np.mean(corr2_out) < np.mean(np.diag(corr2))
    assert 3 * np.mean(corr3_out) < np.mean(np.diag(corr3))
    assert 3 * np.mean(corr4_out) < np.mean(np.diag(corr4))
    assert int1.tuning_data[0].shape == (6, int1.n_v)
    assert int2.tuning_data[0].shape == (6, int2.n_v)
    assert int1.shared_response.shape == (int1.n_t, 6)
    assert X_pred.shape == X_test.shape


def test_int_with_searchlight():
    X_train, X_test, S_true_first_part, S_true_second_part, Ts = generate_dummy_signal(
        n_s=5,
        n_t=50,
        n_v=300,
        S_std=1,
        T_std=1,
        latent_dim=6,
        SNR=100,
        generative_method="custom",
        seed=0,
    )
    searchlights, dists = generate_dummy_searchlights(
        n_searchlights=10, n_v=300, radius=5, seed=0
    )
    from fmralign.hyperalignment.correlation import (
        tuning_correlation,
        stimulus_correlation,
    )

    # Test INT on the two parts of the data (ie different runs of the experiment)
    int1 = INT(n_components=6)
    int2 = INT(n_components=6)
    int1.fit(X_train=X_train, searchlights=searchlights, dists=dists, radius=5)
    int2.fit(X_test, searchlights=searchlights, dists=dists, radius=5)
    X_pred = int1.transform(X_test)

    tuning_data_run_1 = int1.tuning_data
    tuning_data_run_2 = int2.tuning_data
    tuning_data_run_1 = np.array(tuning_data_run_1)
    tuning_data_run_2 = np.array(tuning_data_run_2)

    stimulus_run_1 = int1.shared_response
    stimulus_run_2 = int2.shared_response

    corr1 = tuning_correlation(tuning_data_run_1, tuning_data_run_2)
    corr2 = stimulus_correlation(stimulus_run_1.T, S_true_first_part.T)
    corr3 = stimulus_correlation(stimulus_run_2.T, S_true_second_part.T)
    corr4 = tuning_correlation(X_pred, X_test)

    # Check that predicted components have the same shape as original data

    # Check that the correlation between the two parts of the data is high
    corr1_out = corr1 - np.diag(corr1)
    corr2_out = corr2 - np.diag(corr2)
    corr3_out = corr3 - np.diag(corr3)
    corr4_out = corr4 - np.diag(corr4)
    assert 3 * np.mean(corr1_out) < np.mean(np.diag(corr1))
    assert 3 * np.mean(corr2_out) < np.mean(np.diag(corr2))
    assert 3 * np.mean(corr3_out) < np.mean(np.diag(corr3))
    assert 3 * np.mean(corr4_out) < np.mean(np.diag(corr4))
    assert int1.tuning_data[0].shape == (6, int1.n_v)
    assert int2.tuning_data[0].shape == (6, int2.n_v)
    assert int1.shared_response.shape == (int1.n_t, 6)
    assert X_pred.shape == X_test.shape
