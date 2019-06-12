import numpy as np
from pqueens.main import main
import pytest
import pickle

def test_multivariate_gaussian_metropolis_hastings(tmpdir):
    """ Test case for metropolis hastings iterator """
    arguments = ['--input=pqueens/tests/function_tests/input_files/multivariate_gaussian_metropolis_hastings_multiple_chains.json',
                 '--output='+str(tmpdir)]
    main(arguments)
    result_file = str(tmpdir)+'/'+'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # note that the analytical solution would be:
    # posterior mean: [0.29378531 - 1.97175141]
    # posterior cov: [[0.42937853 0.00282486] [0.00282486 0.00988701]]
    # however, we only have a very inaccurate approximation here:

    np.testing.assert_allclose(results['mean'], np.array([[+1.594133780727028, -1.7692878004798742],
                                                          [+1.6535640118392139, -1.9897871149738702]]))
    np.testing.assert_allclose(results['cov'], np.array([[[+0.19563501424553886, -0.024419993216563844],
                                                          [-0.024419993216563844, +0.0030482072495906613]],
                                                         [[+0.1534354286238552, +0.04940051998889705],
                                                          [+0.04940051998889705, +0.023747509034181782]]]
                                                        )
                           )
