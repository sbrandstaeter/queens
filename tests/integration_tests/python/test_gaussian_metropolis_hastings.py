"""TODO_doc."""

import pytest
from mock import patch

from queens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from queens.main import run
from queens.utils import injector
from queens.utils.io_utils import load_result


def test_gaussian_metropolis_hastings(
    inputdir, tmp_path, target_density_gaussian_1d, _create_experimental_data_gaussian
):
    """Test case for Metropolis Hastings iterator."""
    template = inputdir / "metropolis_hastings_gaussian.yml"
    experimental_data_path = tmp_path
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = tmp_path / "gaussian_metropolis_hastings_realiz.yml"
    injector.inject(dir_dict, template, input_file)

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
    assert results['mean'] == pytest.approx(1.046641592648936)
    assert results['var'] == pytest.approx(0.3190199514534667)
