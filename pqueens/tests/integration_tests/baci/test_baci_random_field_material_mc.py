import os
import pickle

import numpy as np
import pytest
from pqueens.main import main
from pqueens.utils import injector
from pqueens.utils.run_subprocess import run_subprocess


@pytest.mark.integration_tests_baci
def test_write_random_material_to_dat(
    inputdir, tmpdir, third_party_inputs, baci_link_paths, expected_result
):
    # generate json input file from template
    third_party_input_file = os.path.join(
        third_party_inputs, "baci_input_files", "coarse_plate_dirichlet_template.dat"
    )

    new_dat_file_path = os.path.join(tmpdir, "coarse_plate_dirichlet_template.dat")
    cmd_lst = ['/bin/cp -arfp', third_party_input_file, new_dat_file_path]
    command_string = ' '.join(cmd_lst)
    _, _, _, stderr = run_subprocess(command_string)

    baci_release, post_drt_monitor, _, _ = baci_link_paths

    dir_dict = {
        'experiment_dir': str(tmpdir),
        'baci_input': new_dat_file_path,
        'post_drt_monitor': post_drt_monitor,
        'baci-release': baci_release,
    }
    template = os.path.join(inputdir, "baci_random_field_material_mc_template.json")
    input_file = os.path.join(tmpdir, "baci_write_random_field_material.json")
    injector.inject(dir_dict, template, input_file)

    # get json file as config dictionary
    arguments = ['--input=' + input_file, '--output=' + str(tmpdir)]

    # run a MC simulation with random input for now
    main(arguments)

    # Check if we got the expected results
    experiment_name = "baci_write_random_field_to_dat"
    result_file_name = experiment_name + ".pickle"

    result_file = os.path.join(str(tmpdir), result_file_name)
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # assert statements  # TODO check mean and variance displacement at a point for 3 samples
    np.testing.assert_array_almost_equal(
        results['raw_output_data']['mean'], expected_result, decimal=2
    )


@pytest.fixture()
def expected_result():
    result = np.array([[0.751], [0.84], [0.638]]).reshape(3, 1)
    return result
