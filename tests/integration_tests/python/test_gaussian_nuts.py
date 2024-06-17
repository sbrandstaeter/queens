"""Test NUTS Iterator."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mock import patch

from queens.main import run
from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood
from queens.utils import injector
from queens.utils.io_utils import load_result


def test_gaussian_nuts(
    inputdir, tmp_path, target_density_gaussian_2d_with_grad, _create_experimental_data
):
    """Test case for nuts iterator."""
    template = inputdir / "nuts_gaussian.yml"
    experimental_data_path = tmp_path
    dir_dict = {"experimental_data_path": experimental_data_path}
    input_file = tmp_path / "gaussian_nuts_realiz.yml"
    injector.inject(dir_dict, template, input_file)
    with patch.object(
        GaussianLikelihood, "evaluate_and_gradient", target_density_gaussian_2d_with_grad
    ):
        run(Path(input_file), Path(tmp_path))

    results = load_result(tmp_path / 'xxx.pickle')

    assert results['mean'].mean(axis=0) == pytest.approx(
        np.array([-0.2868793496608573, 0.6474274597130008])
    )
    assert results['var'].mean(axis=0) == pytest.approx([0.08396277217936474, 0.10836256575521087])


@pytest.fixture(name="_create_experimental_data")
def fixture_create_experimental_data(tmp_path):
    """Generate 2 samples from the same gaussian."""
    samples = np.array([0, 0]).flatten()

    # write the data to a csv file in tmp_path
    data_dict = {'y_obs': samples}
    experimental_data_path = tmp_path / 'experimental_data.csv'
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
