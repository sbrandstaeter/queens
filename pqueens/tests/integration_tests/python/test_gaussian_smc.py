import os
import pickle
import pandas as pd
from pqueens.utils import injector
import numpy as np
import pytest
from mock import patch

# fmt: off
from pqueens.tests.integration_tests.example_simulator_functions\
    .gaussian_logpdf\
    import (
    standard_normal,
)
from pqueens.tests.integration_tests.example_simulator_functions\
    .gaussian_logpdf\
    import (
    gaussian_logpdf,
)
# fmt: on
from pqueens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from pqueens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from pqueens.main import main


def test_gaussian_smc(inputdir, tmpdir, dummy_data):
    """ Test Sequential Monte Carlo with univariate Gaussian. """
    template = os.path.join(inputdir, "gaussian_smc.json")
    experimental_data_path = tmpdir
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = os.path.join(tmpdir, "gaussian_smc_realiz.json")
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
    # posterior mean: [1.]
    # posterior var: [0.5]
    # posterior std: [0.70710678]
    # however, we only have a very inaccurate approximation here:
    np.testing.assert_almost_equal(results['mean'], np.array([[0.93548976354251]]), decimal=1)
    np.testing.assert_almost_equal(results['var'], np.array([[0.7216833388663654]]), decimal=1)


def target_density(self, samples):
    samples = np.atleast_2d(samples)

    log_lik = []
    for x in samples:
        log_lik.append(gaussian_logpdf(x))

    log_likelihood = np.atleast_2d(np.array(log_lik).flatten()).T

    return log_likelihood


@pytest.fixture()
def dummy_data(tmpdir):
    # generate 10 samples from the same gaussian
    samples = standard_normal.draw(10).flatten()

    # evaluate the gaussian pdf for these 1000 samples
    pdf = []
    for x in samples:
        pdf.append(gaussian_logpdf(x))

    pdf = np.array(pdf).flatten()

    # write the data to a csv file in tmpdir
    data_dict = {'y_obs': pdf}
    experimental_data_path = os.path.join(tmpdir, 'experimental_data.csv')
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)