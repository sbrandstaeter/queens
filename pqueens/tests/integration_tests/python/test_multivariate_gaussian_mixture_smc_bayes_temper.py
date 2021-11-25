import os
import pickle
import pandas as pd
import numpy as np
import pytest
from mock import patch
from pqueens.utils import injector

# fmt: off
from pqueens.tests.integration_tests.example_simulator_functions\
    .multivariate_gaussian_mixture_logpdf\
    import (
    gaussian1,
)
from pqueens.tests.integration_tests.example_simulator_functions\
    .multivariate_gaussian_mixture_logpdf\
    import (
    gaussian_mixture_logpdf,
)
# fmt: on
from pqueens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from pqueens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from pqueens.main import main


def test_multivariate_gaussian_mixture_smc_bayes_temper(inputdir, tmpdir, dummy_data):
    """ Test SMC with a multivariate Gaussian mixture (multimodal). """
    template = os.path.join(inputdir, "multivariate_gaussian_mixture_smc_bayes_temper.json")
    experimental_data_path = tmpdir
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = os.path.join(tmpdir, "multivariate_gaussian_mixture_smc_bayes_temper_realiz.json")
    injector.inject(dir_dict, template, input_file)
    arguments = [
        '--input=' + input_file,
        '--output=' + str(tmpdir),
    ]

    # mock methods related to likelihood
    with patch.object(SequentialMonteCarloIterator, "eval_log_likelihood", target_density):
        with patch.object(MetropolisHastingsIterator, "eval_log_likelihood", target_density):
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


def target_density(self, samples):
    samples = np.atleast_2d(samples)
    x1_vec = samples[:, 0]
    x2_vec = samples[:, 1]
    x3_vec = samples[:, 2]
    x4_vec = samples[:, 3]

    log_lik = []
    for x1, x2, x3, x4 in zip(x1_vec, x2_vec, x3_vec, x4_vec):
        log_lik.append(gaussian_mixture_logpdf(x1, x2, x3, x4))

    log_likelihood = np.atleast_2d(np.array(log_lik)).T

    return log_likelihood


@pytest.fixture()
def dummy_data(tmpdir):
    # generate 10 samples from the same gaussian
    samples = gaussian1.draw(10)
    x1_vec = samples[:, 0]
    x2_vec = samples[:, 1]
    x3_vec = samples[:, 2]
    x4_vec = samples[:, 3]

    # evaluate the gaussian pdf for these 1000 samples
    pdf = []
    for x1, x2, x3, x4 in zip(x1_vec, x2_vec, x3_vec, x4_vec):
        pdf.append(gaussian_mixture_logpdf(x1, x2, x3, x4))

    pdf = np.array(pdf)

    # write the data to a csv file in tmpdir
    data_dict = {'y_obs': pdf}
    experimental_data_path = os.path.join(tmpdir, 'experimental_data.csv')
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)