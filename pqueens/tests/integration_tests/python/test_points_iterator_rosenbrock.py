"""Test points iterator."""

import pickle

import numpy as np
import pytest

from pqueens import run
from pqueens.tests.integration_tests.example_simulator_functions.rosenbrock60 import rosenbrock60
from pqueens.utils.injector import inject


def test_points_iterator(inputdir, tmp_path, inputs, expected_results):
    """Integration test for the points iterator."""
    template = inputdir / 'points_iterator_rosenbrock_template.yml'
    input_file = tmp_path / 'points_iterator_rosenbrock.yml'

    inject(inputs, template, input_file)

    run(input_file, tmp_path)

    result_file = tmp_path / 'points_iterator_rosenbrock.pickle'

    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    np.testing.assert_array_equal(
        results["output"]["mean"],
        expected_results,
    )


def test_points_iterator_failure(inputdir, tmp_path):
    """Test failure of the points iterator."""
    template = inputdir / 'points_iterator_rosenbrock_template.yml'
    input_file = tmp_path / 'points_iterator_rosenbrock_failure.yml'

    inputs = {"x1": [1], "x2": [1, 2]}

    inject(inputs, template, input_file)

    with pytest.raises(
        ValueError, match="Non-matching number of points for the different parameters: x1: 1, x2: 2"
    ):
        run(input_file, tmp_path)


@pytest.fixture(name="inputs")
def fixture_inputs():
    """Input fixtures."""
    return {"x1": [1, 2], "x2": [3, 4]}


@pytest.fixture(name="expected_results")
def fixture_expected_results(inputs):
    """Expected results fixture."""
    input_as_array = inputs.copy()
    input_as_array["x1"] = np.array(input_as_array["x1"]).reshape(-1, 1)
    input_as_array["x2"] = np.array(input_as_array["x2"]).reshape(-1, 1)
    return rosenbrock60(**input_as_array)
