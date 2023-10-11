"""Test HMC Sampler."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mock import patch

from pqueens import run
from pqueens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood
from pqueens.tests.integration_tests.example_simulator_functions.gaussian_logpdf import (
    gaussian_2d_logpdf,
)
from pqueens.utils import injector


def test_gaussian_hmc(inputdir, tmp_path, dummy_data):
    """Test case for hmc iterator."""
    template = inputdir / "hmc_gaussian.yml"
    experimental_data_path = tmp_path
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = tmp_path / "gaussian_hmc_realiz.yml"
    injector.inject(dir_dict, template, input_file)

    with patch.object(GaussianLikelihood, "evaluate_and_gradient", target_density):
        run(Path(input_file), Path(tmp_path))

    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    assert results['mean'].mean(axis=0) == pytest.approx(
        np.array([0.19363280864587615, -1.1303341362165935])
    )
    assert results['var'].mean(axis=0) == pytest.approx([0, 0])


def target_density(self, samples):
    """Patch likelihood."""
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_2d_logpdf(samples).flatten()

    cov = [[1.0, 0.5], [0.5, 1.0]]
    cov_inverse = np.linalg.inv(cov)
    gradient = -np.dot(cov_inverse, samples.T).T

    return log_likelihood, gradient


@pytest.fixture(name="dummy_data")
def fixture_dummy_data(tmp_path):
    """Generate 2 samples from the same gaussian."""
    samples = np.array([0, 0]).flatten()

    # write the data to a csv file in tmp_path
    data_dict = {'y_obs': samples}
    experimental_data_path = tmp_path / 'experimental_data.csv'
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
