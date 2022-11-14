import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from pqueens import run


def test_optimization_lsq_parabula(inputdir, tmpdir):
    """Test special case for optimization iterator with least squares.

    Special case: 1 unknown and 1 residual
    """
    run(Path(os.path.join(inputdir, 'optimization_lsq_parabula.yml')), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'ParabulaResLSQ.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    np.testing.assert_allclose(results.x, np.array([+0.3]))
    np.testing.assert_allclose(results.fun, np.array([+0.0]))
