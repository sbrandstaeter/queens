import numpy as np
from pqueens.main import main
import pytest
import pickle

def test_multivariate_gaussian_metropolis_hastings(tmpdir):
    """ Test case for metropolis hastings iterator """
    arguments = ['--input=pqueens/tests/function_tests/input_files/multivariate_gaussian_metropolis_hastings.json',
                 '--output='+str(tmpdir)]

    main(arguments)
    result_file = str(tmpdir)+'/'+'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # note that the analytical solution would be:
    # posterior mean: [0.29378531 -1.97175141]
    # posterior cov: [[0.42937853 0.00282486] [0.00282486 0.00988701]]
    # however, we only have a very inaccurate approximation here:

    np.testing.assert_allclose(results['mean'], np.array([[-0.6617825437718209, -0.8987041259572249]]))
    np.testing.assert_allclose(results['cov'], np.array([[[0.4530521248041719, -0.10537928263612602],
                                                          [-0.10537928263612602, 0.037124714876106704]]]))
