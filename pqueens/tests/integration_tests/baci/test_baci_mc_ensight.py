"""Test BACI run."""

import pickle

import numpy as np
import pytest

from pqueens import run
from pqueens.utils import injector


def test_baci_mc_ensight(
    inputdir, tmp_path, third_party_inputs, baci_link_paths, expected_mean, expected_var
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
    np.testing.assert_array_almost_equal(results['mean'], expected_mean, decimal=6)
    np.testing.assert_array_almost_equal(results['var'], expected_var, decimal=6)


@pytest.fixture(name="expected_mean")
def fixture_expected_mean():
    """Expected result."""
    result = np.array(
        [
            [0.0041549, 0.00138497, -0.00961201],
            [0.00138497, 0.00323159, -0.00961201],
            [0.00230828, 0.00323159, -0.00961201],
            [0.0041549, 0.00230828, -0.00961201],
            [0.00138497, 0.0041549, -0.00961201],
            [0.0041549, 0.00323159, -0.00961201],
            [0.00230828, 0.0041549, -0.00961201],
            [0.0041549, 0.0041549, -0.00961201],
            [0.00138497, 0.00138497, -0.00961201],
            [0.00323159, 0.00138497, -0.00961201],
            [0.00138497, 0.00230828, -0.00961201],
            [0.00230828, 0.00138497, -0.00961201],
            [0.00323159, 0.00230828, -0.00961201],
            [0.00230828, 0.00230828, -0.00961201],
            [0.00323159, 0.00323159, -0.00961201],
            [0.00323159, 0.0041549, -0.00961201],
        ]
    )
    return result


@pytest.fixture(name="expected_var")
def name_expected_var():
    """Expected variance."""
    result = np.array(
        [
            [3.19513506e-07, 3.55014593e-08, 2.94994460e-07],
            [3.55014593e-08, 1.93285820e-07, 2.94994460e-07],
            [9.86153027e-08, 1.93285820e-07, 2.94994460e-07],
            [3.19513506e-07, 9.86153027e-08, 2.94994460e-07],
            [3.55014593e-08, 3.19513506e-07, 2.94994460e-07],
            [3.19513506e-07, 1.93285820e-07, 2.94994460e-07],
            [9.86153027e-08, 3.19513506e-07, 2.94994460e-07],
            [3.19513506e-07, 3.19513506e-07, 2.94994460e-07],
            [3.55014593e-08, 3.55014593e-08, 2.94994460e-07],
            [1.93285820e-07, 3.55014593e-08, 2.94994460e-07],
            [3.55014593e-08, 9.86153027e-08, 2.94994460e-07],
            [9.86153027e-08, 3.55014593e-08, 2.94994460e-07],
            [1.93285820e-07, 9.86153027e-08, 2.94994460e-07],
            [9.86153027e-08, 9.86153027e-08, 2.94994460e-07],
            [1.93285820e-07, 1.93285820e-07, 2.94994460e-07],
            [1.93285820e-07, 3.19513506e-07, 2.94994460e-07],
        ]
    )
    return result
