"""Test suite for integration tests of optimization iterator.

Based on the Rosenbrock test function.
"""
import pickle

import numpy as np
import pytest

from pqueens import run
from pqueens.utils import injector


@pytest.fixture(name="algorithm", params=['NELDER-MEAD', 'POWELL', 'CG', 'BFGS', 'L-BFGS-B', 'TNC'])
def fixture_algorithm(request):
    """TODO_doc."""
    return request.param


def test_optimization_rosenbrock(inputdir, tmp_path, algorithm):
    """Test different solution algorithms in optimization iterator."""
    template = inputdir / 'optimization_rosenbrock_template.yml'
    input_file = tmp_path / 'rosenbrock_opt.yml'

    algorithm_dict = {'algorithm': algorithm}

    injector.inject(algorithm_dict, template, input_file)

    run(input_file, tmp_path)

    result_file = tmp_path / 'Rosenbrock.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    np.testing.assert_allclose(results.x, np.array([+1.0, +1.0]), rtol=1.0e-3)
    np.testing.assert_allclose(results.fun, np.array(+0.0), atol=1.0e-07)
