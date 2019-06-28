import numpy as np
from pqueens.main import main
import pytest
import pickle

def test_multivariate_gaussian_metropolis_hastings_multiple_chains(tmpdir):
    """ Test case for metropolis hastings iterator """
    arguments = ['--input=pqueens/tests/function_tests/input_files/multivariate_gaussian_metropolis_hastings_multiple_chains.json',
                 '--output='+str(tmpdir)]
    main(arguments)
    result_file = str(tmpdir)+'/'+'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # note that the analytical solution would be:
    # posterior mean: [0.29378531 -1.97175141]
    # posterior cov: [[0.42937853 0.00282486] [0.00282486 0.00988701]]
    # however, we only have a very inaccurate approximation here:

    np.testing.assert_allclose(results['mean'],
                               np.array([[1.9538477050387937, -1.980155948698723],
                                         [-0.024456540006756778, -1.9558862932202299],
                                         [0.8620026644863327, -1.8385635263327393]]
                                                         )
                               )
    np.testing.assert_allclose(results['cov'],
                               np.array([[[0.15127359388133552, 0.07282531084034029],
                                          [0.07282531084034029, 0.05171405742642703]],
                                         [[0.17850797646369507, -0.012342979562824052],
                                          [-0.012342979562824052, 0.0023510303586270057]],
                                         [[0.0019646760257596243, 0.002417903725921208],
                                          [0.002417903725921208, 0.002975685737073754]]]
                                                        )
                           )

