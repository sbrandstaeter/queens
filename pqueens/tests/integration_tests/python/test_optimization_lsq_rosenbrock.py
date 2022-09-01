import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from pqueens import run


@pytest.mark.integration_tests
def test_optimization_lsq_rosenbrock(inputdir, tmpdir):
    """Test case for optimization iterator with least squares."""
    run(Path(os.path.join(inputdir, 'optimization_lsq_rosenbrock.json')), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'ResRosenbrockLSQ.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    np.testing.assert_allclose(results.x, np.array([+1.0, +1.0]))
    np.testing.assert_allclose(results.fun, np.array([+0.0, +0.0]))
