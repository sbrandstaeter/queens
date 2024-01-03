"""TODO_doc."""

import pickle

import numpy as np
import pytest

from queens.main import run


def test_neural_network_gauss_bayesian_branin(inputdir, tmp_path, expected_mean, expected_var):
    """Test case for Bayesian neural network model."""
    run(inputdir / 'neural_network_gauss_bayesian_branin.yml', tmp_path)

    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["mean"], expected_mean, decimal=4
    )
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["variance"], expected_var, decimal=2
    )


@pytest.fixture(name="expected_mean")
def fixture_expected_mean():
    """TODO_doc."""
    mean = np.array(
        [
            [
                65.37786,
                65.44934,
                44.39922,
                57.19025,
                64.86770,
                65.44933,
                65.44935,
                65.44935,
                65.44862,
                22.31277,
            ]
        ]
    )
    return mean.T


@pytest.fixture(name="expected_var")
def fixture_expected_var():
    """TODO_doc."""
    var = np.array(
        [
            [
                3.31274,
                3.31792,
                2.04469,
                2.76017,
                3.27650,
                3.31792,
                3.31792,
                3.31792,
                3.31787,
                1.08863,
            ]
        ]
    )
    return var.T
