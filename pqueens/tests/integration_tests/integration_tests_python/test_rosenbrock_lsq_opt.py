import os
import pickle

import numpy as np
import pytest

from pqueens.main import main


def test_rosenbrock_lsq_opt(inputdir, tmpdir):
    """ Test case for optimization iterator with least squares. """
    arguments = [
        '--input=' + os.path.join(inputdir, 'rosenbrock_lsq_opt.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'ResRosenbrockLSQ.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    np.testing.assert_allclose(results.x, np.array([+1.0, +1.0]))
    np.testing.assert_allclose(results.fun, np.array([+0.0, +0.0]))
