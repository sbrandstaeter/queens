"""TODO_doc."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from pqueens import run
from pqueens.utils import injector


def test_write_random_material_to_dat(
    inputdir, tmp_path, third_party_inputs, baci_link_paths, expected_result
):
    """TODO_doc."""
    # generate json input file from template
    third_party_input_file = Path(
        third_party_inputs, "baci_input_files", "coarse_plate_dirichlet_template.dat"
    )

    dat_file_preprocessed = tmp_path.joinpath("coarse_plate_dirichlet_template_preprocessed.dat")

    baci_release, post_drt_monitor, _, _ = baci_link_paths

    dir_dict = {
        'baci_input': third_party_input_file,
        'baci_input_preprocessed': dat_file_preprocessed,
        'post_drt_monitor': post_drt_monitor,
        'baci-release': baci_release,
    }
    template = inputdir.joinpath("baci_random_field_material_mc_template.yml")
    input_file = tmp_path.joinpath("baci_write_random_field_material.yml")
    injector.inject(dir_dict, template, input_file)

    # run a MC simulation with random input for now
    run(input_file, tmp_path)

    # Check if we got the expected results
    experiment_name = "baci_write_random_field_to_dat"
    result_file_name = experiment_name + ".pickle"

    result_file = tmp_path.joinpath(result_file_name)
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # assert statements
    # TODO check mean and variance displacement at a point for 3 samples
    # TODO the tolerances on this test are to coarse
    np.testing.assert_array_almost_equal(
        results['raw_output_data']['mean'], expected_result, decimal=1
    )


@pytest.fixture()
def expected_result():
    """TODO_doc."""
    result = np.array([[0.79857, 0.73370, 0.71603]]).reshape(3, 1)
    return result
