import os
import pickle
import numpy as np
import pytest

from pqueens.main import main


def test_branin_gp_surrogate(inputdir, tmpdir, expected_pdf):
    """ Test case for GP based surrogate model """
    arguments = [
        '--input=' + os.path.join(inputdir, 'branin_gp_surrogate.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    np.testing.assert_array_almost_equal(results["pdf_estimate"]["mean"], expected_pdf, decimal=4)


@pytest.fixture()
def expected_pdf():
    pdf = [
        0.00055841,
        0.00112721,
        0.00219813,
        0.00368771,
        0.0052036,
        0.00580986,
        0.00481893,
        0.00334127,
        0.00199375,
        0.00110298,
    ]
    return np.array(pdf)
