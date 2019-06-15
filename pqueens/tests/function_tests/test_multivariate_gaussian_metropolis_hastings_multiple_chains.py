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

    np.testing.assert_allclose(results['mean'], np.array([[-1.4755414998480934, -1.1477192680388755],
                                                          [-1.5753796208998556, -1.2264743587381162]]
                                                         )
                               )
    np.testing.assert_allclose(results['cov'], np.array([[[+0.002377693372328929, +0.009612679187985102],
                                                          [+0.009612679187985102, +0.03886270712888995]],
                                                         [[+0.8296151928349861,   +0.10063229603621723],
                                                          [+0.10063229603621723,  +0.03476356435097165]]]
                                                        )
                           )
