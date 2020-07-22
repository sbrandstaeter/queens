import os
import pickle

import pytest

from pqueens.main import main


def test_gaussian_smc(inputdir, tmpdir):
    """ Test Sequential Monte Carlo with univariate Gaussian. """
    arguments = [
        '--input=' + os.path.join(inputdir, 'gaussian_smc.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # note that the analytical solution would be:
    # posterior mean: [1.]
    # posterior var: [0.5]
    # posterior std: [0.70710678]
    # however, we only have a very inaccurate approximation here:
    assert results['mean'] == pytest.approx(0.93548976354251)
    assert results['var'] == pytest.approx(0.7216833388663654)
