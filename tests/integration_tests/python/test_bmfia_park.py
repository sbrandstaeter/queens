"""Integration tests for the BMFIA."""
# pylint: disable=invalid-name
import pickle
from pathlib import Path

import numpy as np
import pytest

from queens.main import run
from queens.utils import injector


@pytest.mark.max_time_for_test(30)
def test_bmfia_smc_park(
    inputdir,
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    expected_samples,
    expected_weights,
):
    """Integration test for BMFIA.

    Integration test for bayesian multi-fidelity inverse analysis
    (bmfia) using the park91 function.
    """
    # generate yml input file from template
    template = inputdir / 'bmfia_smc_park.yml'
    experimental_data_path = tmp_path
    dir_dict = {"experimental_data_path": experimental_data_path, "plot_dir": tmp_path}
    input_file = tmp_path / 'smc_mf_park_realization.yml'
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(Path(input_file), Path(tmp_path))

    # get the results of the QUEENS run
    result_file = tmp_path / 'smc_park_mf.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    samples = results['raw_output_data']['particles'].squeeze()
    weights = results['raw_output_data']['weights'].squeeze()

    # some tests / asserts here
    np.testing.assert_array_almost_equal(samples, expected_samples, decimal=5)
    np.testing.assert_array_almost_equal(weights.flatten(), expected_weights.flatten(), decimal=5)


@pytest.fixture(name="expected_samples")
def fixture_expected_samples():
    """Fixture for expected SMC samples."""
    samples = np.array(
        [
            [0.51711296, 0.55200585],
            [0.4996905, 0.6673229],
            [0.48662203, 0.68802404],
            [0.49806929, 0.66276797],
            [0.49706481, 0.68586978],
            [0.50424704, 0.65139028],
            [0.51437955, 0.57678317],
            [0.51275639, 0.58981357],
            [0.50163956, 0.65389397],
            [0.52127371, 0.61237995],
        ]
    )

    return samples


@pytest.fixture(name="expected_weights")
def fixture_expected_weights():
    """Fixture for expected SMC weights."""
    weights = np.array(
        [
            0.00183521,
            0.11284748,
            0.16210619,
            0.07066473,
            0.10163831,
            0.09845534,
            0.10742886,
            0.15461861,
            0.09222745,
            0.0981778,
        ]
    )
    return weights


@pytest.mark.max_time_for_test(20)
def test_bmfia_rpvi_gp_park(
    inputdir,
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    expected_variational_mean,
    expected_variational_cov,
):
    """Integration test for BMFIA.

    Integration test for bayesian multi-fidelity inverse analysis
    (bmfia) using the park91 function.
    """
    template = inputdir / 'bmfia_rpvi_park_gp_template.yml'
    experimental_data_path = tmp_path
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": tmp_path,
    }
    input_file = tmp_path / 'bmfia_rpvi_park.yml'
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(Path(input_file), Path(tmp_path))

    # get the results of the QUEENS run
    result_file = tmp_path / 'bmfia_rpvi_park.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    variational_mean = results['variational_distribution']['mean']
    variational_cov = results['variational_distribution']['covariance']

    # some tests / asserts here
    np.testing.assert_array_almost_equal(variational_mean, expected_variational_mean, decimal=2)
    np.testing.assert_array_almost_equal(variational_cov, expected_variational_cov, decimal=2)


@pytest.fixture(name="expected_variational_mean")
def fixture_expected_variational_mean():
    """Fixture for expected variational_mean."""
    exp_var_mean = np.array([0.53399236, 0.52731554]).reshape(-1, 1)

    return exp_var_mean


@pytest.fixture(name="expected_variational_cov")
def fixture_expected_variational_cov():
    """Fixture for expected variational covariance."""
    exp_var_cov = np.array([[0.00142648, 0.0], [0.0, 0.00347234]])
    return exp_var_cov


def test_bmfia_rpvi_NN_park(
    inputdir,
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    expected_variational_mean_nn,
    expected_variational_cov_nn,
):
    """Integration test for BMFIA.

    Integration test for bayesian multi-fidelity inverse analysis
    (bmfia) using the park91 function.
    """
    template = inputdir / 'bmfia_rpvi_park_NN_template.yml'
    experimental_data_path = tmp_path
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": tmp_path,
    }
    input_file = tmp_path / 'bmfia_rpvi_park.yml'
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(Path(input_file), Path(tmp_path))

    # get the results of the QUEENS run
    result_file = tmp_path / 'bmfia_rpvi_park.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    variational_mean = results['variational_distribution']['mean']
    variational_cov = results['variational_distribution']['covariance']

    # some tests / asserts here
    np.testing.assert_array_almost_equal(variational_mean, expected_variational_mean_nn, decimal=1)
    np.testing.assert_array_almost_equal(variational_cov, expected_variational_cov_nn, decimal=1)


@pytest.fixture(name="expected_variational_mean_nn")
def fixture_expected_variational_mean_nn():
    """Fixture for expected variational_mean."""
    exp_var_mean = np.array([0.19221321, 0.33134219]).reshape(-1, 1)

    return exp_var_mean


@pytest.fixture(name="expected_variational_cov_nn")
def fixture_expected_variational_cov_nn():
    """Fixture for expected variational covariance."""
    exp_var_cov = np.array([[0.01245263, 0.0], [0.0, 0.01393423]])
    return exp_var_cov
