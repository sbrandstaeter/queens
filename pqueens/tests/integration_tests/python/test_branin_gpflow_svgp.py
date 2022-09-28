import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from pqueens import run


def test_branin_gpflow_svgp(inputdir, tmpdir, expected_mean, expected_var):
    """Test case for GPflow based SVGP model."""
    run(Path(os.path.join(inputdir, 'gpflow_svgp_surrogate_branin.json')), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["mean"], expected_mean, decimal=4
    )
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["variance"], expected_var, decimal=2
    )


@pytest.fixture()
def expected_mean():
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


@pytest.fixture()
def expected_var():
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
