"""TODO_doc."""

import numpy as np
from mock import patch

from queens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from queens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from queens.main import run
from queens.utils import injector
from queens.utils.io_utils import load_result


def test_gaussian_smc(
    inputdir, tmp_path, target_density_gaussian_1d, _create_experimental_data_gaussian
):
    """Test Sequential Monte Carlo with univariate Gaussian."""
    template = inputdir / "smc_gaussian.yml"
    experimental_data_path = tmp_path  # pylint: disable=duplicate-code
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = tmp_path / "gaussian_smc_realiz.yml"
    injector.inject(dir_dict, template, input_file)
    # mock methods related to likelihood
    with patch.object(
        SequentialMonteCarloIterator, "eval_log_likelihood", target_density_gaussian_1d
    ):
        with patch.object(
            MetropolisHastingsIterator, "eval_log_likelihood", target_density_gaussian_1d
        ):
            run(input_file, tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')

    # note that the analytical solution would be:
    # posterior mean: [1.]
    # posterior var: [0.5]
    # posterior std: [0.70710678]
    # however, we only have a very inaccurate approximation here:
    np.testing.assert_almost_equal(results['mean'], np.array([[0.93548976]]), decimal=7)
    np.testing.assert_almost_equal(results['var'], np.array([[0.72168334]]), decimal=7)
