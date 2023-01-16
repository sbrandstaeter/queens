"""TODO_doc."""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mock import patch

from pqueens import run
from pqueens.interfaces import from_config_create_interface
from pqueens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from pqueens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from pqueens.tests.integration_tests.example_simulator_functions.gaussian_logpdf import (
    gaussian_4d,
    gaussian_4d_logpdf,
)
from pqueens.utils import injector


def test_smc_generic_temper_multivariate_gaussian(inputdir, tmpdir, dummy_data):
    """Test SMC with a multivariate Gaussian and generic tempering."""
    template = os.path.join(inputdir, "smc_generic_temper_multivariate_gaussian.json")
    experimental_data_path = tmpdir
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = os.path.join(tmpdir, "multivariate_gaussian_smc_generic_temper_realiz.json")
    injector.inject(dir_dict, template, input_file)
    # mock methods related to likelihood
    with patch.object(SequentialMonteCarloIterator, "eval_log_likelihood", target_density):
        with patch.object(MetropolisHastingsIterator, "eval_log_likelihood", target_density):
            run(Path(input_file), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'GaussSMCGenTemp.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # note that the analytical solution can be found in multivariate_gaussian_4D_logpdf
    # we only have a very inaccurate approximation here:
    np.testing.assert_array_almost_equal(
        results['mean'], np.array([[0.884713, 2.903405, -3.112647, 1.56134]]), decimal=5
    )

    np.testing.assert_almost_equal(
        results['var'], np.array([[3.255066, 4.143380, 1.838545, 2.834356]]), decimal=5
    )

    np.testing.assert_almost_equal(
        results['cov'],
        np.array(
            [
                [
                    [3.255066, 1.781563, 0.313565, -0.090972],
                    [1.781563, 4.143380, 0.779616, 1.704881],
                    [0.313565, 0.779616, 1.838545, 0.630236],
                    [-0.090972, 1.704881, 0.630236, 2.834356],
                ]
            ]
        ),
        decimal=5,
    )


def target_density(self, samples):
    """TODO_doc."""
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_4d_logpdf(samples).reshape(-1, 1)

    return log_likelihood


@pytest.fixture()
def dummy_data(tmpdir):
    """TODO_doc."""
    # generate 10 samples from the same gaussian
    samples = gaussian_4d.draw(10)
    pdf = gaussian_4d_logpdf(samples)

    # write the data to a csv file in tmpdir
    data_dict = {'y_obs': pdf}
    experimental_data_path = os.path.join(tmpdir, 'experimental_data.csv')
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
