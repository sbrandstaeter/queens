"""Test Sobol indices estimation for Ishigami function."""

import numpy as np

from queens.main import run
from queens.utils.io_utils import load_result
from test_utils.integration_tests import assert_sobol_index_iterator_results


def test_sobol_indices_ishigami(inputdir, tmp_path):
    """Test case for Salib based Saltelli iterator."""
    run(inputdir / 'sobol_indices_ishigami.yml', tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')

    expected_result = {}

    expected_result["S1"] = np.array([0.12572757495660558, 0.3888444532476749, -0.1701023677236496])

    expected_result["S1_conf"] = np.array(
        [0.3935803586836114, 0.6623091120357786, 0.2372589075839736]
    )

    expected_result["ST"] = np.array([0.32520201992825987, 0.5263552164769918, 0.1289289258091274])

    expected_result["ST_conf"] = np.array(
        [0.24575185898081872, 0.5535870474744364, 0.15792828597131078]
    )

    expected_result["S2"] = np.array(
        [
            [np.nan, 0.6350854922111611, 1.0749774123116016],
            [np.nan, np.nan, 0.32907368546743065],
            [np.nan, np.nan, np.nan],
        ]
    )

    expected_result["S2_conf"] = np.array(
        [
            [np.nan, 0.840605849268133, 1.2064077218919202],
            [np.nan, np.nan, 0.5803799668636836],
            [np.nan, np.nan, np.nan],
        ]
    )

    assert_sobol_index_iterator_results(results, expected_result)
