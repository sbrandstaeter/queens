"""TODO_doc."""

import numpy as np
import pytest

from queens.main import run
from queens.utils.io_utils import load_result


@pytest.mark.max_time_for_test(60)
def test_branin_gpflow_svgp(inputdir, tmp_path, expected_mean, expected_var):
    """Test case for GPflow based SVGP model."""
    run(inputdir / 'gpflow_svgp_surrogate_branin.yml', tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')

    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["result"], expected_mean, decimal=4
    )
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["variance"], expected_var, decimal=2
    )


@pytest.fixture(name="expected_mean")
def fixture_expected_mean():
    """TODO_doc."""
    mean = np.array(
        [
            [181.62057979],
            [37.95455295],
            [47.86422341],
            [32.47391656],
            [23.99246991],
            [167.32578661],
            [106.07427664],
            [92.93591941],
            [50.72976800],
            [22.10505115],
        ]
    )
    return mean


@pytest.fixture(name="expected_var")
def fixture_expected_var():
    """TODO_doc."""
    var = np.array(
        [
            [4.62061],
            [1.38456],
            [0.96146],
            [0.20286],
            [0.34231],
            [1.03465],
            [0.24111],
            [0.40275],
            [0.22169],
            [0.58071],
        ]
    )
    return var
