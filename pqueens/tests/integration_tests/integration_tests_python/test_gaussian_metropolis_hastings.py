import os
import pickle

import pytest

from pqueens.main import main


def test_gaussian_metropolis_hastings(inputdir, tmpdir):
    """ Test case for metropolis hastings iterator """
    arguments = [
        '--input=' + os.path.join(inputdir, 'gaussian_metropolis_hastings.json'),
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
    assert results['mean'] == pytest.approx(1.046641592648936)
    assert results['var'] == pytest.approx(0.3190199514534667)
