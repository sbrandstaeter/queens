import pickle

import numpy as np
import pytest

from pqueens.main import main


def test_multivariate_gaussian_mixture_smc(tmpdir):
    """ Test SMC with a multivariate Gaussian mixture (multimodal). """
    arguments = [
        '--input=pqueens/tests/function_tests/input_files/multivariate_gaussian_mixture_smc.json',
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
        results['mean'], np.array([[0.17395033, 0.14691232, 0.10394709, 0.133549]])
    )

    np.testing.assert_allclose(
        results['var'], np.array([[0.21775397, 0.188199, 0.20164435, 0.30171133]])
    )

    np.testing.assert_allclose(
        results['cov'],
        np.array(
            [
                [
                    [0.21775397, 0.19686943, 0.20260172, 0.25429943],
                    [0.19686943, 0.188199, 0.18582571, 0.23619157],
                    [0.20260172, 0.18582571, 0.20164435, 0.23668967],
                    [0.25429943, 0.23619157, 0.23668967, 0.30171133],
                ]
            ]
        ),
    )
