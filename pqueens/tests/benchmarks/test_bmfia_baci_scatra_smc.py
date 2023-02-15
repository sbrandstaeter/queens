"""TODO_doc."""

import pickle
from pathlib import Path

import numpy as np
import pytest

import pqueens.visualization.bmfia_visualization as qvis
from pqueens import run
from pqueens.utils import injector


def test_bmfia_baci_scatra_smc(inputdir, tmp_path, third_party_inputs, config_dir):
    """TODO_doc: add a one-line description.

    Integration test for smc with a simple diffusion problem (scatra) in
    BACI.
    """
    # generate yaml input file from template
    third_party_input_file_hf = third_party_inputs / "baci_input_files/diffusion_coarse.dat"
    third_party_input_file_lf = third_party_inputs / "baci_input_files/diffusion_very_coarse.dat"

    baci_release = config_dir / "baci-release"
    post_drt_ensight = config_dir / "post_drt_ensight"

    # ----- generate yaml input file from template -----
    # template for actual smc evaluation
    template = inputdir / 'bmfia_scatra_baci_template_smc_gp_precompiled_copy.yml'

    experimental_data_path = third_party_inputs / "csv_files/scatra_baci"
    plot_dir = tmp_path
    dir_dict = {
        'experimental_data_path': experimental_data_path,
        'baci_hf_input': third_party_input_file_hf,
        'baci_lf_input': third_party_input_file_lf,
        'baci-release': baci_release,
        'post_drt_ensight': post_drt_ensight,
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

    samples = results['raw_output_data']['particles'].squeeze()
    weights = results['raw_output_data']['weights'].squeeze()

    dim_labels_lst = ['x_s', 'y_s']
    qvis.bmfia_visualization_instance.plot_posterior_from_samples(samples, weights, dim_labels_lst)

    np.testing.assert_array_almost_equal(weights, expected_weights, decimal=5)
    np.testing.assert_array_almost_equal(samples, expected_samples, decimal=5)


@pytest.fixture()
def expected_weights():
    """TODO_doc."""
    weights = 1
    return weights


@pytest.fixture()
def expected_samples():
    """TODO_doc."""
    samples = 1
    return samples
