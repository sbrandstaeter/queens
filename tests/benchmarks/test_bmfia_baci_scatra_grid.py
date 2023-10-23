"""Benchmark test for BMFIA using a grid iterator."""

import pickle

import numpy as np
import pytest

import queens.visualization.bmfia_visualization as qvis
from queens.main import run
from queens.utils import injector


def test_bmfia_baci_scatra_smc(inputdir, tmp_path, third_party_inputs, config_dir):
    """Integration test for smc with a simple diffusion problem in BACI."""
    # generate json input file from template
    third_party_input_file_hf = third_party_inputs / "baci" / "diffusion_coarse.dat"
    third_party_input_file_lf = third_party_inputs / "baci" / "diffusion_very_coarse.dat"

    baci_release = config_dir / "baci-release"
    post_ensight = config_dir / "post_ensight"

    # ----- generate yaml input file from template -----
    # template for actual smc evaluation
    template = inputdir / 'bmfia_scatra_baci_template_grid_gp_precompiled.yml'

    experimental_data_path = third_party_inputs / "csv" / "scatra_baci"
    plot_dir = tmp_path
    dir_dict = {
        'experimental_data_path': experimental_data_path,
        'baci_hf_input': third_party_input_file_hf,
        'baci_lf_input': third_party_input_file_lf,
        'baci-release': baci_release,
        'post_ensight': post_ensight,
        'plot_dir': plot_dir,
    }
    input_file = tmp_path / 'hf_scatra_baci.yml'
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(input_file, tmp_path)

    # get the results of the QUEENS run
    result_file = tmp_path / 'bmfia_baci_scatra_smc.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    samples = results['input_data'].squeeze()
    weights = results['raw_output_data'].squeeze()
    sum_weights = np.sum(weights)
    weights = weights / sum_weights

    dim_labels_lst = ['x_s', 'y_s']
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
