import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from pqueens import run


def test_optimization_lsq_rosenbrock_1d(inputdir, tmpdir):
    """Test special case for optimization iterator with least squares.

    Special case: 1 unknown but 2 residuals
    """
    run(Path(Path(inputdir, 'optimization_lsq_rosenbrock_1d.json')), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'ResRosenbrock1DLSQ.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    np.testing.assert_allclose(results.x, np.array([+1.0]))
    np.testing.assert_allclose(results.fun, np.array([+0.0, +0.0]))
