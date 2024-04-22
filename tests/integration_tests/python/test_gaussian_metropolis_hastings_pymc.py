"""Test PyMC MH Sampler."""

from pathlib import Path

import numpy as np
import pytest
from mock import patch

from queens.example_simulator_functions.gaussian_logpdf import gaussian_2d_logpdf
from queens.main import run
from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood
from queens.utils import injector
from queens.utils.io_utils import load_result


def test_gaussian_mh(inputdir, tmp_path, _create_experimental_data_zero):
    """Test case for mh iterator."""
    template = inputdir / "mh_gaussian.yml"
    experimental_data_path = tmp_path
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = tmp_path / "gaussian_mh_realiz.yml"
    injector.inject(dir_dict, template, input_file)
    with patch.object(GaussianLikelihood, "evaluate", target_density):
        run(Path(input_file), Path(tmp_path))

    results = load_result(tmp_path / 'xxx.pickle')

    assert results['mean'].mean(axis=0) == pytest.approx(
        np.array([-0.5680310153118374, 0.9247536392514567])
    )
    assert results['var'].mean(axis=0) == pytest.approx([0.13601070852470507, 0.6672200465857734])


def target_density(self, samples):  # pylint: disable=unused-argument
    """Patch likelihood."""
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_2d_logpdf(samples).flatten()

    return log_likelihood
