import pickle

import numpy as np
import pytest

from pqueens.main import main


def test_multivariate_gaussian_smc_generic_temper(tmpdir):
    """ Test SMC with a multivariate Gaussian and generic tempering. """
    arguments = [
        '--input=pqueens/tests/function_tests/input_files/multivariate_gaussian_smc_generic_temper.json',
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'GaussSMCGenTemp.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # note that the analytical solution can be found in multivariate_gaussian_4D_logpdf
    # we only have a very inaccurate approximation here:
    np.testing.assert_allclose(
        results['mean'], np.array([[-0.36912335, 1.58058413, -2.09632999, 0.15901674]])
    )

    np.testing.assert_allclose(
        results['var'], np.array([[0.37561109, 1.74547135, 1.12387232, 1.21785044]])
    )

    np.testing.assert_allclose(
        results['cov'],
        np.array(
            [
                [
                    [0.37561109, 0.35547374, 0.46541726, -0.15728409],
                    [0.35547374, 1.74547135, 0.51962791, 0.16952631],
                    [0.46541726, 0.51962791, 1.12387232, -0.04186952],
                    [-0.15728409, 0.16952631, -0.04186952, 1.21785044],
                ]
            ]
        ),
    )
