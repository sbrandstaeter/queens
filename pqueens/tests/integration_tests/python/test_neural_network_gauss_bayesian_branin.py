"""TODO_doc."""

import pickle

import numpy as np
import pytest

from pqueens import run


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
def expected_mean_fixture():
    """TODO_doc."""
    mean = np.array(
        [
            [
                65.3779063,
                65.44938406,
                44.40122105,
                57.19055642,
                64.8677122,
                65.44937106,
                65.44938677,
                65.44938672,
                65.44865757,
                22.31121954,
            ]
        ]
    )
    return mean.T


@pytest.fixture(name="expected_var")
def expected_var_fixture():
    """TODO_doc."""
    var = np.array(
        [
            [
                3.31275871,
                3.31793473,
                2.04479705,
                2.7602009,
                3.27651827,
                3.31793384,
                3.31793497,
                3.31793497,
                3.31788235,
                1.08857106,
            ]
        ]
    )
    return var.T
