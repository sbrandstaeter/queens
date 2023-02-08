"""TODO_doc."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mock import patch

from pqueens import run
from pqueens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from pqueens.tests.integration_tests.example_simulator_functions.gaussian_logpdf import (
    gaussian_1d_logpdf,
    standard_normal,
)
from pqueens.utils import injector


def test_gaussian_metropolis_hastings(inputdir, tmp_path, dummy_data):
    """Test case for Metropolis Hastings iterator."""
    template = inputdir / "metropolis_hastings_gaussian.yml"
    experimental_data_path = tmp_path
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = tmp_path / "gaussian_metropolis_hastings_realiz.yml"
    injector.inject(dir_dict, template, input_file)
    with patch.object(MetropolisHastingsIterator, "eval_log_likelihood", target_density):
        run(input_file, tmp_path)

    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # note that the analytical solution would be:
    # posterior mean: [1.]
    # posterior var: [0.5]
    # posterior std: [0.70710678]
    # however, we only have a very inaccurate approximation here:
    assert results['mean'] == pytest.approx(1.046641592648936)
    assert results['var'] == pytest.approx(0.3190199514534667)


def target_density(self, samples):
    """TODO_doc."""
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_1d_logpdf(samples).reshape(-1, 1)

    return log_likelihood


@pytest.fixture()
def dummy_data(tmp_path):
    """TODO_doc."""
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
