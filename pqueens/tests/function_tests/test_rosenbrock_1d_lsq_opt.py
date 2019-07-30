import pickle

import numpy as np
import pytest

from pqueens.main import main


def test_rosenbrock_1d_lsq_opt(tmpdir):
    """
    Test special case for optimization iterator with least squares.

    Special case: 1 unknown but 2 residuals
    """
    arguments = ['--input=pqueens/tests/function_tests/input_files/rosenbrock_1d_lsq_opt.json',
                 '--output='+str(tmpdir)]

    main(arguments)
    result_file = str(tmpdir)+'/'+'ResRosenbrock1DLSQ.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    np.testing.assert_allclose(results.x, np.array([+1.0]))
    np.testing.assert_allclose(results.fun, np.array([+0.0, +0.0]))
