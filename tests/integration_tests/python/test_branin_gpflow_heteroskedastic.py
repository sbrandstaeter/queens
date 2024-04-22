"""TODO_doc."""

import numpy as np
import pytest

from queens.main import run
from queens.utils.io_utils import load_result


@pytest.mark.max_time_for_test(30)
def test_branin_gpflow_heteroskedastic(inputdir, tmp_path, expected_mean, expected_var):
    """Test case for GPflow based heteroskedastic model."""
    run(inputdir / 'gp_heteroskedastic_surrogate_branin.yml', tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')

    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["result"], expected_mean, decimal=2
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
                5.12898,
                4.07712,
                10.22693,
                2.55123,
                4.56184,
                2.45215,
                2.56100,
                3.32164,
                7.84209,
                6.96919,
            ]
        ]
    ).T
    return mean


@pytest.fixture(name="expected_var")
def fixture_expected_var():
    """TODO_doc."""
    var = np.array(
        [
            [
                1057.66078,
                4802.57196,
                1298.08163,
                1217.39827,
                456.70756,
                13143.74176,
                8244.52203,
                21364.59699,
                877.14343,
                207.58535,
            ]
        ]
    ).T
    return var
