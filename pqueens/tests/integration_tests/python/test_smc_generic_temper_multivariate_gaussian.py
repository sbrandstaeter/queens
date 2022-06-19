import os
import pickle

import numpy as np
import pandas as pd
import pytest
from mock import patch

from pqueens.interfaces import from_config_create_interface
from pqueens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from pqueens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from pqueens.main import main
from pqueens.tests.integration_tests.example_simulator_functions.gaussian_logpdf import (
    gaussian_4d,
    gaussian_4d_logpdf,
)
from pqueens.utils import injector


@pytest.mark.integration_tests
def test_smc_generic_temper_multivariate_gaussian(inputdir, tmpdir, dummy_data):
    """Test SMC with a multivariate Gaussian and generic tempering."""
    template = os.path.join(inputdir, "smc_generic_temper_multivariate_gaussian.json")
    experimental_data_path = tmpdir
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = os.path.join(tmpdir, "multivariate_gaussian_smc_generic_temper_realiz.json")
    injector.inject(dir_dict, template, input_file)
    arguments = [
        '--input=' + input_file,
        '--output=' + str(tmpdir),
    ]
    # mock methods related to likelihood
    with patch.object(SequentialMonteCarloIterator, "eval_log_likelihood", target_density):
        with patch.object(MetropolisHastingsIterator, "eval_log_likelihood", target_density):
            main(arguments)

    result_file = str(tmpdir) + '/' + 'GaussSMCGenTemp.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # note that the analytical solution can be found in multivariate_gaussian_4D_logpdf
    # we only have a very inaccurate approximation here:
    np.testing.assert_allclose(
        results['mean'], np.array([[0.60398075, 2.52994971, -3.18073644, 1.33579982]])
    )

    np.testing.assert_allclose(
        results['var'], np.array([[2.86233632, 4.19022998, 2.37767776, 2.87405536]])
    )

    np.testing.assert_allclose(
        results['cov'],
        np.array(
            [
                [
                    [2.86233632, 1.47874047, 0.41524659, -0.00426986],
                    [1.47874047, 4.19022998, 0.88684972, 1.53565931],
                    [0.41524659, 0.88684972, 2.37767776, 0.60398836],
                    [-0.00426986, 1.53565931, 0.60398836, 2.87405536],
                ]
            ]
        ),
    )


def target_density(self, samples):
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_4d_logpdf(samples).reshape(-1, 1)

    return log_likelihood


@pytest.fixture()
def dummy_data(tmpdir):
    # generate 10 samples from the same gaussian
    samples = gaussian_4d.draw(10)
    pdf = gaussian_4d_logpdf(samples)

    # write the data to a csv file in tmpdir
    data_dict = {'y_obs': pdf}
    experimental_data_path = os.path.join(tmpdir, 'experimental_data.csv')
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
