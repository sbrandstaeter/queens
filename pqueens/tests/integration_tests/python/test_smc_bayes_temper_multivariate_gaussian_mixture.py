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
from pqueens.tests.integration_tests.example_simulator_functions.gaussian_mixture_logpdf import (
    gaussian_component_1,
    gaussian_mixture_4d_logpdf,
)
from pqueens.utils import injector


def test_smc_bayes_temper_multivariate_gaussian_mixture(inputdir, tmpdir, dummy_data):
    """Test SMC with a multivariate Gaussian mixture (multimodal)."""
    template = os.path.join(inputdir, "smc_bayes_temper_multivariate_gaussian_mixture.json")
    experimental_data_path = tmpdir
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = os.path.join(tmpdir, "multivariate_gaussian_mixture_smc_bayes_temper_realiz.json")
    injector.inject(dir_dict, template, input_file)

    # mock methods related to likelihood
    with patch.object(SequentialMonteCarloIterator, "eval_log_likelihood", target_density):
        with patch.object(MetropolisHastingsIterator, "eval_log_likelihood", target_density):
            run(Path(input_file), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # note that the analytical solution would be:
    # posterior mean: [-0.4 -0.4 -0.4 -0.4]
    # posterior var: [0.1, 0.1, 0.1, 0.1]
    # however, we only have a very inaccurate approximation here:
    np.testing.assert_almost_equal(
        results['mean'], np.array([[0.23384, 0.21806, 0.24079, 0.24528]]), decimal=5
    )

    np.testing.assert_almost_equal(
        results['var'], np.array([[0.30894, 0.15192, 0.19782, 0.18781]]), decimal=5
    )

    np.testing.assert_almost_equal(
        results['cov'],
        np.array(
            [
                [
                    [0.30894, 0.21080, 0.24623, 0.23590],
                    [0.21080, 0.15192, 0.17009, 0.15951],
                    [0.24623, 0.17009, 0.19782, 0.18695],
                    [0.23590, 0.15951, 0.18695, 0.18781],
                ]
            ]
        ),
        decimal=5,
    )


def target_density(self, samples):
    """TODO_doc."""
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_mixture_4d_logpdf(samples).reshape(-1, 1)

    return log_likelihood


@pytest.fixture()
def dummy_data(tmpdir):
    """TODO_doc."""
    # generate 10 samples from the same gaussian
    samples = gaussian_component_1.draw(10)
    pdf = gaussian_mixture_4d_logpdf(samples)

    pdf = np.array(pdf)

    # write the data to a csv file in tmpdir
    data_dict = {'y_obs': pdf}
    experimental_data_path = os.path.join(tmpdir, 'experimental_data.csv')
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
