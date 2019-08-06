import pickle

import numpy as np
import pytest

from pqueens.main import main


def test_parabula_lsq_opt(tmpdir):
    """
    Test special case for optimization iterator with least squares.

    Special case: 1 unknown and 1 residual
    """
    arguments = ['--input=pqueens/tests/function_tests/input_files/parabula_lsq_opt.json',
                 '--output='+str(tmpdir)]

    main(arguments)
    result_file = str(tmpdir)+'/'+'ParabulaResLSQ.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    np.testing.assert_allclose(results.x, np.array([+0.3]))
    np.testing.assert_allclose(results.fun, np.array([+0.0]))
