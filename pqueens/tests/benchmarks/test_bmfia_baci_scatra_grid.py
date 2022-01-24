"""Benchmark test for BMFIA using a grid iterator."""

import os
import pickle

import numpy as np
import pytest

import pqueens.visualization.bmfia_visualization as qvis
from pqueens.main import main
from pqueens.utils import injector


@pytest.mark.benchmark
def test_bmfia_baci_scatra_smc(inputdir, tmpdir, third_party_inputs, config_dir):
    """Integration test for smc with a simple diffusion problem in BACI."""
    # generate json input file from template
    third_party_input_file_hf = os.path.join(
        third_party_inputs, "baci_input_files", "diffusion_coarse.dat"
    )
    third_party_input_file_lf = os.path.join(
        third_party_inputs, "baci_input_files", "diffusion_very_coarse.dat"
    )

    baci_release = os.path.join(config_dir, "baci-release")
    post_drt_ensight = os.path.join(config_dir, "post_drt_ensight")

    # ----- generate json input file from template -----
    # template for actual smc evaluation
    template = os.path.join(inputdir, 'bmfia_scatra_baci_template_grid_gp_precompiled.json')

    experimental_data_path = os.path.join(third_party_inputs, "csv_files", "scatra_baci")
    plot_dir = tmpdir
    dir_dict = {
        'experimental_data_path': experimental_data_path,
        'experiment_dir': str(tmpdir),
        'baci_hf_input': third_party_input_file_hf,
        'baci_lf_input': third_party_input_file_lf,
        'baci-release': baci_release,
        'post_drt_ensight': post_drt_ensight,
        'plot_dir': plot_dir,
    }
    input_file = os.path.join(tmpdir, 'hf_scatra_baci.json')
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    arguments = [
        '--input=' + input_file,
        '--output=' + str(tmpdir),
    ]

    # actual main call of smc
    main(arguments)

    # get the results of the QUEENS run
    result_file = os.path.join(tmpdir, 'bmfia_baci_scatra_smc.pickle')
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


@pytest.fixture()
def expected_weights():
    """Dummy weights."""
    weights = 1
    return weights


@pytest.fixture()
def expected_samples():
    """Dummy samples."""
    samples = 1
    return samples