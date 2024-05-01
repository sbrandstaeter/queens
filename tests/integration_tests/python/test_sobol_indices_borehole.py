"""Test Sobol indices estimation for borehole function."""

import numpy as np

from queens.main import run
from queens.utils.io_utils import load_result


def test_sobol_indices_borehole(inputdir, tmp_path):
    """Test case for Sobol Index iterator."""
    run(inputdir / 'sobol_indices_borehole.yml', tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')

    expected_first_order_indices = np.array(
        [
            0.8275788005095177,
            3.626326582692376e-05,
            1.7993448562887368e-09,
            0.04082350205109995,
            -1.0853339811788176e-05,
            0.0427473897346278,
            0.038941629762778956,
            0.009001905983634081,
        ]
    )

    np.testing.assert_allclose(results["sensitivity_indices"]["S1"], expected_first_order_indices)
