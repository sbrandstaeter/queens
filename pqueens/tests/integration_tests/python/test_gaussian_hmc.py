"""Test HMC Sampler."""
import os
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


def test_gaussian_hmc(inputdir, tmpdir, dummy_data):
    """Test case for hmc iterator."""
    template = os.path.join(inputdir, "hmc_gaussian.json")
    experimental_data_path = tmpdir
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = os.path.join(tmpdir, "gaussian_hmc_realiz.json")
    injector.inject(dir_dict, template, input_file)
    with patch.object(GaussianLikelihood, "evaluate", target_density):
        run(Path(input_file), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    assert results['mean'].mean(axis=0) == pytest.approx(
        np.array([0.19363280864587615, -1.1303341362165935])
    )
    assert results['var'].mean(axis=0) == pytest.approx([0, 0])


def target_density(self, samples, gradient_bool):
    """Patch likelihood."""
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_2d_logpdf(samples).flatten()

    cov = [[1.0, 0.5], [0.5, 1.0]]
    cov_inverse = np.linalg.inv(cov)
    gradient = -np.dot(cov_inverse, samples.T).T

    return (log_likelihood, gradient)


@pytest.fixture()
def dummy_data(tmpdir):
    """Generate 2 samples from the same gaussian."""
    samples = np.array([0, 0]).flatten()

    # write the data to a csv file in tmpdir
    data_dict = {'y_obs': samples}
    experimental_data_path = os.path.join(tmpdir, 'experimental_data.csv')
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
