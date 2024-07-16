"""Benchmark test for BMFIA using a grid iterator."""

import numpy as np
import pytest

import queens.visualization.bmfia_visualization as qvis
from queens.main import run
from queens.utils import injector
from queens.utils.io_utils import load_result


def test_bmfia_baci_scatra_smc(inputdir, tmp_path, paths_dictionary):
    """Integration test for smc with a simple diffusion problem in BACI."""
    # generate yaml input file from template
    # template for actual smc evaluation
    template = inputdir / "bmfia_scatra_baci_template_grid_gp_precompiled.yml"
    input_file = tmp_path / "hf_scatra_baci.yml"
    injector.inject(paths_dictionary, template, input_file)

    # run the main routine of QUEENS
    run(input_file, tmp_path)

    # Load results
    result_file = tmp_path / "dummy_experiment_name.pickle"
    results = load_result(result_file)

    samples = results["input_data"].squeeze()
    weights = results["raw_output_data"].squeeze()
    sum_weights = np.sum(weights)
    weights = weights / sum_weights

    dim_labels_lst = ["x_s", "y_s"]
    qvis.bmfia_visualization_instance.plot_posterior_from_samples(samples, weights, dim_labels_lst)

    # np.testing.assert_array_almost_equal(weights, expected_weights, decimal=5)
    # np.testing.assert_array_almost_equal(samples, expected_samples, decimal=5)


@pytest.fixture(name="expected_weights")
def fixture_expected_weights():
    """Dummy weights."""
    weights = 1
    return weights


@pytest.fixture(name="expected_samples")
def fixture_expected_samples():
    """Dummy samples."""
    samples = 1
    return samples
