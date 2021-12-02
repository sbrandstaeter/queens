"""
Test suite for integration tests of optimization iterator

based on the Rosenbrock test function
"""
import os
import pickle

import numpy as np
import pytest

from pqueens.main import main
from pqueens.utils import injector


@pytest.fixture(params=['NELDER-MEAD', 'POWELL', 'CG', 'BFGS', 'L-BFGS-B', 'TNC'])
def algorithm(request):

    return request.param


@pytest.mark.integration_tests
def test_rosenbrock_opt(inputdir, tmpdir, algorithm):
    """ Test different solution algorithms in optimization iterator. """

    template = os.path.join(inputdir, 'rosenbrock_opt_template.json')
    input_file = str(tmpdir) + 'rosenbrock_opt.json'

    algorithm_dict = {'algorithm': algorithm}

    injector.inject(algorithm_dict, template, input_file)

    arguments = ['--input=' + input_file, '--output=' + str(tmpdir)]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'Rosenbrock.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    np.testing.assert_allclose(results.x, np.array([+1.0, +1.0]), rtol=1.0e-3)
    np.testing.assert_allclose(results.fun, np.array(+0.0), atol=1.0e-07)
