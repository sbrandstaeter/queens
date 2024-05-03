"""Test HMC Sampler."""

from pathlib import Path

import numpy as np
import pytest
from mock import patch

from queens.main import run
from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood
from queens.utils import injector
from queens.utils.io_utils import load_result


def test_gaussian_hmc(
    inputdir, tmp_path, target_density_gaussian_2d_with_grad, _create_experimental_data_zero
):
    """Test case for hmc iterator."""
    template = inputdir / "hmc_gaussian.yml"
    # pylint: disable=duplicate-code
    experimental_data_path = tmp_path
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = tmp_path / "gaussian_hmc_realiz.yml"
    injector.inject(dir_dict, template, input_file)

    with patch.object(
        GaussianLikelihood, "evaluate_and_gradient", target_density_gaussian_2d_with_grad
    ):
        run(Path(input_file), Path(tmp_path))

    results = load_result(tmp_path / 'xxx.pickle')

    assert results['mean'].mean(axis=0) == pytest.approx(
        np.array([0.19363280864587615, -1.1303341362165935])
    )
    assert results['var'].mean(axis=0) == pytest.approx([0, 0])
