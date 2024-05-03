"""TODO_doc."""

import numpy as np
import pytest

from queens.main import run
from queens.utils.io_utils import load_result
from test_utils.integration_tests import assert_monte_carlo_iterator_results


def test_neural_network_gauss_bayesian_branin(inputdir, tmp_path, expected_mean, expected_var):
    """Test case for Bayesian neural network model."""
    run(inputdir / 'neural_network_gauss_bayesian_branin.yml', tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')
    assert_monte_carlo_iterator_results(results, expected_mean, expected_var)


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
