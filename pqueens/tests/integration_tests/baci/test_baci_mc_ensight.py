"""Test BACI run."""

import pickle

import numpy as np

from pqueens.main import run
from pqueens.utils import injector


def test_baci_mc_ensight(
    inputdir,
    tmp_path,
    third_party_inputs,
    baci_link_paths,
    baci_example_expected_mean,
    baci_example_expected_var,
    baci_example_expected_output,
):
    """Test simple BACI run."""
    # generate json input file from template
    third_party_input_file = (
        third_party_inputs / "baci_input_files" / "meshtying3D_patch_lin_duallagr_new_struct.dat"
    )
    baci_release, post_ensight, _ = baci_link_paths
    dir_dict = {
        'baci_input': third_party_input_file,
        'post_ensight': post_ensight,
        'baci_release': baci_release,
    }
    template = inputdir / "baci_mc_ensight_template.yml"
    input_file = tmp_path / "baci_mc_ensight.yml"
    injector.inject(dir_dict, template, input_file)

    # get json file as config dictionary
    run(input_file, tmp_path)

    # run a MC simulation with random input for now

    # Check if we got the expected results
    experiment_name = "baci_mc_ensight"
    result_file_name = experiment_name + ".pickle"

    result_file = tmp_path / result_file_name
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # assert statements
    np.testing.assert_array_almost_equal(results['mean'], baci_example_expected_mean, decimal=6)
    np.testing.assert_array_almost_equal(results['var'], baci_example_expected_var, decimal=6)
    np.testing.assert_array_almost_equal(
        results['raw_output_data']['mean'], baci_example_expected_output, decimal=6
    )
