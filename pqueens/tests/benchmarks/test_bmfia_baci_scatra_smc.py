import os
import pickle
from pathlib import Path

import numpy as np
import pytest

import pqueens.visualization.bmfia_visualization as qvis
from pqueens import run
from pqueens.utils import injector


def test_bmfia_baci_scatra_smc(inputdir, tmpdir, third_party_inputs, config_dir):
    """Integration test for smc with a simple diffusion problem (scatra) in
    baci."""
    # generate json input file from template
    third_party_input_file_hf = Path(third_party_inputs, "baci_input_files", "diffusion_coarse.dat")
    third_party_input_file_lf = Path(
        third_party_inputs, "baci_input_files", "diffusion_very_coarse.dat"
    )

    baci_release = Path(config_dir, "baci-release")
    post_drt_ensight = Path(config_dir, "post_drt_ensight")

    # ----- generate json input file from template -----
    # template for actual smc evaluation
    template = Path(inputdir, 'bmfia_scatra_baci_template_smc_gp_precompiled_copy.json')

    experimental_data_path = Path(third_party_inputs, "csv_files", "scatra_baci")
    plot_dir = tmpdir
    dir_dict = {
        'experimental_data_path': experimental_data_path,
        'baci_hf_input': third_party_input_file_hf,
        'baci_lf_input': third_party_input_file_lf,
        'baci-release': baci_release,
        'post_drt_ensight': post_drt_ensight,
        'plot_dir': plot_dir,
    }
    input_file = Path(tmpdir, 'hf_scatra_baci.json')
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(Path(input_file), Path(tmpdir))

    # get the results of the QUEENS run
    result_file = Path(tmpdir, 'bmfia_baci_scatra_smc.pickle')
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
    weights = 1
    return weights


@pytest.fixture()
def expected_samples():
    samples = 1
    return samples
