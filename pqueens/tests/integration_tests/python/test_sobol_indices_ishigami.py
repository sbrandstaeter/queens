"""Test Sobol indices estimation for Ishigami function."""
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from pqueens import run


def test_sobol_indices_ishigami(inputdir, tmpdir):
    """Test case for Salib based Saltelli iterator."""
    run(Path(os.path.join(inputdir, 'sobol_indices_ishigami.json')), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    expected_result = dict()

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

    np.testing.assert_allclose(results["sensitivity_indices"]["S1"], expected_result["S1"])
    np.testing.assert_allclose(
        results["sensitivity_indices"]["S1_conf"], expected_result["S1_conf"]
    )

    np.testing.assert_allclose(results["sensitivity_indices"]["ST"], expected_result["ST"])
    np.testing.assert_allclose(
        results["sensitivity_indices"]["ST_conf"], expected_result["ST_conf"]
    )

    np.testing.assert_allclose(results["sensitivity_indices"]["S2"], expected_result["S2"])
    np.testing.assert_allclose(
        results["sensitivity_indices"]["S2_conf"], expected_result["S2_conf"]
    )
