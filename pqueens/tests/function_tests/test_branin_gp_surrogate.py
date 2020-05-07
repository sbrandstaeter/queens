import os
import pickle

import pytest

from pqueens.main import main


def test_branin_gp_surrogate(inputdir, tmpdir):
    """ Test case for GP based surrogate model """
    arguments = [
        '--input=' + os.path.join(inputdir, 'branin_gp_surrogate.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["pdf_estimate"]["mean"][1] == pytest.approx(0.016946978829947043, abs=0.5e-6)
