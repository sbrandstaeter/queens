"""Test suite for integration tests of optimization iterator.

Based on the Rosenbrock test function.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from pqueens import run
from pqueens.utils import injector


@pytest.fixture(params=['NELDER-MEAD', 'POWELL', 'CG', 'BFGS', 'L-BFGS-B', 'TNC'])
def algorithm(request):
    """TODO_doc."""
    return request.param


def test_optimization_rosenbrock(inputdir, tmpdir, algorithm):
    """Test different solution algorithms in optimization iterator."""
    template = os.path.join(inputdir, 'optimization_rosenbrock_template.yml')
    input_file = str(tmpdir) + 'rosenbrock_opt.yml'

    algorithm_dict = {'algorithm': algorithm}

    injector.inject(algorithm_dict, template, input_file)

    run(Path(input_file), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'Rosenbrock.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    np.testing.assert_allclose(results.x, np.array([+1.0, +1.0]), rtol=1.0e-3)
    np.testing.assert_allclose(results.fun, np.array(+0.0), atol=1.0e-07)
