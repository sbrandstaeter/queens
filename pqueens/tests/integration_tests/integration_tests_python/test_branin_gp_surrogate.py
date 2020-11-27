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
        0.00274925,
        0.0082232,
        0.01554549,
        0.01509496,
        0.00887515,
        0.00362993,
        0.00143253,
        0.00123129,
        0.00100905,
        0.0003595,
    ]
    return np.array(pdf)
