import os
import pickle

import numpy as np
import pytest

from pqueens.main import main


def test_multivariate_gaussian_mixture_smc_bayes_temper(inputdir, tmpdir):
    """ Test SMC with a multivariate Gaussian mixture (multimodal). """
    arguments = [
        '--input=' + os.path.join(inputdir, 'multivariate_gaussian_mixture_smc_bayes_temper.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # note that the analytical solution would be:
    # posterior mean: [-0.4 -0.4 -0.4 -0.4]
    # posterior var: [0.1, 0.1, 0.1, 0.1]
    # however, we only have a very inaccurate approximation here:
    np.testing.assert_allclose(
        results['mean'], np.array([[-0.53293528, -0.47730905, -0.50474165, -0.51011509]])
    )

    np.testing.assert_allclose(
        results['var'],
        np.array(
            [
                [
                    0.007316406587829176,
                    0.0029133218428340833,
                    0.005518659870323712,
                    0.00818496997630454,
                ]
            ]
        ),
    )

    np.testing.assert_allclose(
        results['cov'],
        np.array(
            [
                [
                    [
                        0.007316406587829175,
                        -0.0007885967653077669,
                        -0.0013053250444199829,
                        -0.0002919686494523325,
                    ],
                    [
                        -0.0007885967653077669,
                        0.002913321842834083,
                        0.0025587041396506224,
                        -0.001115272621522347,
                    ],
                    [
                        -0.0013053250444199829,
                        0.0025587041396506224,
                        0.005518659870323712,
                        -0.00029490035214039926,
                    ],
                    [
                        -0.0002919686494523325,
                        -0.001115272621522347,
                        -0.00029490035214039926,
                        0.008184969976304537,
                    ],
                ]
            ]
        ),
    )
