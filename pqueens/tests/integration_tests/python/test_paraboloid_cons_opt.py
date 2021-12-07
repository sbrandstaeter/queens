import os
import pickle

import numpy as np
import pytest

from pqueens.main import main
from pqueens.utils import injector


@pytest.fixture(params=['COBYLA', 'SLSQP'])
def algorithm(request):

    return request.param


@pytest.mark.integration_tests
def test_paraboloid_cons_opt(inputdir, tmpdir, algorithm):
    """Test different solution algorithms in optimization iterator.

    COBYLA: constained but unbounded
    SLSQP:  constrained and bounded
    """

    template = os.path.join(inputdir, 'paraboloid_opt_template.json')
    input_file = str(tmpdir) + 'paraboloid_opt.json'

    algorithm_dict = {'algorithm': algorithm}

    injector.inject(algorithm_dict, template, input_file)

    arguments = ['--input=' + input_file, '--output=' + str(tmpdir)]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'Paraboloid.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    np.testing.assert_allclose(results.x, np.array([+1.4, +1.7]), rtol=1.0e-4)
    np.testing.assert_allclose(results.fun, np.array(+0.8), atol=1.0e-07)
