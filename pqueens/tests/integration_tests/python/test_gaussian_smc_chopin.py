"""Tests for Chopin SMC module from 'particles'."""

import pickle

import numpy as np
import pandas as pd
import pytest
from mock import patch

from pqueens import run

# fmt: on
from pqueens.iterators.sequential_monte_carlo_chopin import SequentialMonteCarloChopinIterator

# fmt: off
from pqueens.tests.integration_tests.example_simulator_functions.gaussian_logpdf import (
    gaussian_1d_logpdf,
    standard_normal,
)
from pqueens.utils import injector


def test_gaussian_smc_chopin_adaptive_tempering(inputdir, tmp_path, dummy_data):
    """Test Sequential Monte Carlo with univariate Gaussian."""
    template = inputdir / "smc_chopin_gaussian.yml"
    experimental_data_path = tmp_path
    dir_dict = {"experimental_data_path": experimental_data_path, "fk_method": "adaptive_tempering"}
    input_file = tmp_path / "gaussian_smc_realiz.yml"
    injector.inject(dir_dict, template, input_file)
    # mock methods related to likelihood
    with patch.object(SequentialMonteCarloChopinIterator, "eval_log_likelihood", target_density):
        run(input_file, tmp_path)


    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # note that the analytical solution would be:
    # posterior mean: [1.]
    # posterior var: [0.5]
    # posterior std: [0.70710678]
    # however, we only have a very inaccurate approximation here:
    assert np.abs(results['raw_output_data']['mean'] - 1) < 0.2
    assert np.abs(results['raw_output_data']['var'] - 0.5) < 0.2


def target_density(self, samples):
    """Target posterior density."""
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_1d_logpdf(samples).reshape(-1, 1)

    return log_likelihood


@pytest.fixture()
def dummy_data(tmp_path):
    """Fixture for dummy data."""
    # generate 10 samples from the same gaussian
    samples = standard_normal.draw(10).flatten()

    # evaluate the gaussian pdf for these 1000 samples
    pdf = []
    for x in samples:
        pdf.append(gaussian_1d_logpdf(x))

    pdf = np.array(pdf).flatten()

    # write the data to a csv file in tmp_path
    data_dict = {'y_obs': pdf}
    experimental_data_path = tmp_path / 'experimental_data.csv'
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
