"""Test Sobol indices estimation for borehole function."""
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from pqueens import run


def test_sobol_indices_borehole(inputdir, tmpdir):
    """Test case for Sobol Index iterator."""
    run(Path(Path(inputdir, 'sobol_indices_borehole.yml')), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    expected_result_S1 = np.array(
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

    np.testing.assert_allclose(results["sensitivity_indices"]["S1"], expected_result_S1)
