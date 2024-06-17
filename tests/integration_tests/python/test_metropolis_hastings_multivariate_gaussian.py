"""TODO_doc."""

import numpy as np
from mock import patch

# fmt: off
from queens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator

# fmt: on
from queens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from queens.main import run
from queens.utils import injector
from queens.utils.io_utils import load_result


def test_metropolis_hastings_multivariate_gaussian(
    inputdir, tmp_path, target_density_gaussian_2d, _create_experimental_data_gaussian_2d
):
    """Test case for Metropolis Hastings iterator."""
    # pylint: disable=duplicate-code
    template = inputdir / "metropolis_hastings_multivariate_gaussian.yml"
    experimental_data_path = tmp_path
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = tmp_path / "multivariate_gaussian_metropolis_hastings_realiz.yml"
    injector.inject(dir_dict, template, input_file)

    # mock methods related to likelihood
    with patch.object(
        SequentialMonteCarloIterator, "eval_log_likelihood", target_density_gaussian_2d
    ):
        with patch.object(
            MetropolisHastingsIterator, "eval_log_likelihood", target_density_gaussian_2d
        ):
            run(input_file, tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')

    # note that the analytical solution would be:
    # posterior mean: [0.29378531 -1.97175141]
    # posterior cov: [[0.42937853 0.00282486] [0.00282486 0.00988701]]
    # however, we only have a very inaccurate approximation here:

    np.testing.assert_allclose(
        results['mean'], np.array([[0.7240107551260684, -2.045891088599629]])
    )
    np.testing.assert_allclose(
        results['cov'],
        np.array(
            [
                [
                    [0.30698538649168755, -0.027059991075557278],
                    [-0.027059991075557278, 0.004016365725389411],
                ]
            ]
        ),
    )
