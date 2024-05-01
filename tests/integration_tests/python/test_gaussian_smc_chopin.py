"""Tests for Chopin SMC module from 'particles'."""

import numpy as np
from mock import patch

# fmt: on
from queens.iterators.sequential_monte_carlo_chopin import SequentialMonteCarloChopinIterator
from queens.main import run
from queens.utils import injector
from queens.utils.io_utils import load_result


def test_gaussian_smc_chopin_adaptive_tempering(
    inputdir, tmp_path, target_density_gaussian_1d, _create_experimental_data_gaussian
):
    """Test Sequential Monte Carlo with univariate Gaussian."""
    template = inputdir / "smc_chopin_gaussian.yml"
    experimental_data_path = tmp_path  # pylint: disable=duplicate-code
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = tmp_path / "gaussian_smc_realiz.yml"
    injector.inject(dir_dict, template, input_file)
    # mock methods related to likelihood
    with patch.object(
        SequentialMonteCarloChopinIterator, "eval_log_likelihood", target_density_gaussian_1d
    ):
        run(input_file, tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')

    # note that the analytical solution would be:
    # posterior mean: [1.]
    # posterior var: [0.5]
    # posterior std: [0.70710678]
    # however, we only have a very inaccurate approximation here:
    assert np.abs(results['raw_output_data']['mean'] - 1) < 0.2
    assert np.abs(results['raw_output_data']['var'] - 0.5) < 0.2
