"""TODO_doc."""

import pickle

import numpy as np

from queens.main import run


def test_optimization_lsq_rosenbrock_1d(inputdir, tmp_path):
    """Test special case for optimization iterator with least squares.

    Special case: 1 unknown but 2 residuals.
    """
    run(inputdir / 'optimization_lsq_rosenbrock_1d.yml', tmp_path)

    result_file = tmp_path / 'ResRosenbrock1DLSQ.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    np.testing.assert_allclose(results.x, np.array([+1.0]))
    np.testing.assert_allclose(results.fun, np.array([+0.0, +0.0]))
