"""TODO_doc."""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mock import patch

from pqueens import run
from pqueens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from pqueens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from pqueens.tests.integration_tests.example_simulator_functions.gaussian_logpdf import (
    gaussian_2d,
    gaussian_2d_logpdf,
)
from pqueens.utils import injector


def test_metropolis_hastings_multiple_chains_multivariate_gaussian(inputdir, tmpdir, dummy_data):
    """Test case for Metropolis Hastings iterator."""
    template = Path(inputdir, "metropolis_hastings_multiple_chains_multivariate_gaussian.yml")
    experimental_data_path = tmpdir
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = Path(
        tmpdir, "multivariate_gaussian_metropolis_hastings_multiple_chains_realiz.yml"
    )
    injector.inject(dir_dict, template, input_file)
    # mock methods related to likelihood
    with patch.object(SequentialMonteCarloIterator, "eval_log_likelihood", target_density):
        with patch.object(MetropolisHastingsIterator, "eval_log_likelihood", target_density):
            run(Path(input_file), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # note that the analytical solution would be:
    # posterior mean: [0.29378531 -1.97175141]
    # posterior cov: [[0.42937853 0.00282486] [0.00282486 0.00988701]]
    # however, we only have a very inaccurate approximation here:

    np.testing.assert_allclose(
        results['mean'],
        np.array(
            [
                [1.9538477050387937, -1.980155948698723],
                [-0.024456540006756778, -1.9558862932202299],
                [0.8620026644863327, -1.8385635263327393],
            ]
        ),
    )
    np.testing.assert_allclose(
        results['cov'],
        np.array(
            [
                [
                    [0.15127359388133552, 0.07282531084034029],
                    [0.07282531084034029, 0.05171405742642703],
                ],
                [
                    [0.17850797646369507, -0.012342979562824052],
                    [-0.012342979562824052, 0.0023510303586270057],
                ],
                [
                    [0.0019646760257596243, 0.002417903725921208],
                    [0.002417903725921208, 0.002975685737073754],
                ],
            ]
        ),
    )


def target_density(self, samples):
    """TODO_doc."""
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_2d_logpdf(samples).reshape(-1, 1)

    return log_likelihood


@pytest.fixture()
def dummy_data(tmpdir):
    """TODO_doc."""
    # generate 10 samples from the same gaussian
    samples = gaussian_2d.draw(10)
    pdf = gaussian_2d_logpdf(samples)

    # write the data to a csv file in tmpdir
    data_dict = {'y_obs': pdf}
    experimental_data_path = Path(tmpdir, 'experimental_data.csv')
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
