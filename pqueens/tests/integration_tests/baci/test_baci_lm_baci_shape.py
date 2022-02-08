"""Test suite for the heritage baci Levenberg-Marquardt optimizer.

Test local simulations with BACI using a minimal FSI model and the
post_post_baci_shape evaluation and therefore post_drt_ensight post-
processor.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from pqueens.main import main
from pqueens.utils import injector
from pqueens.utils.manage_singularity import hash_files
from pqueens.utils.run_subprocess import run_subprocess


@pytest.fixture(params=[False, True])
def singularity_bool(request):
    """Return boolean to run with or without singularity.

    Args:
        request (SubRequest): true = with singularity; false = without singularity

    Returns:
        request.param (bool): true = with singularity; false = without singularity
    """
    return request.param


@pytest.fixture(scope="session")
def output_directory_forward(tmpdir_factory):
    """Create two temporary output directories for test runs with singularity.

    with singularity (<...>_true) and without singularity (<...>_false)

    Args:
        tmpdir_factory: fixture used to create arbitrary temporary directories

    Returns:
        output_directory_forward (dict): temporary output directories for simulation without and
                                         with singularity
    """
    path_singularity_true = tmpdir_factory.mktemp("test_baci_lm_shape_singularity")
    path_singularity_false = tmpdir_factory.mktemp("test_baci_lm_shape_nosingularity")

    return {True: path_singularity_true, False: path_singularity_false}


@pytest.fixture()
def experiment_directory(output_directory_forward, singularity_bool):
    """Return experiment directory depending on singularity_bool.

    Returns:
        experiment_directory (LocalPath): experiment directory depending on singularity_bool
    """
    return output_directory_forward[singularity_bool]


@pytest.mark.integration_tests_baci
def test_baci_lm_shape(
    inputdir,
    third_party_inputs,
    config_dir,
    baci_link_paths,
    singularity_bool,
    experiment_directory,
):
    """Integration test for the Baci Levenberg Marquardt Iterator with BACI.

    The test runs local native BACI simulations as well as a local
    Singularity based BACI simulations.

    Args:
        inputdir (str): Path to the JSON input file
        third_party_inputs (str): Path to the BACI input files
        config_dir (str): Path to the config directory of QUEENS containing BACI executables
        set_baci_links_for_gitlab_runner (str): Several paths that are needed to build symbolic
                                                links to executables
        singularity_bool (str): String that encodes a boolean that is parsed to the JSON input file
        experiment_directory (LocalPath): experiment directory depending on singularity_bool

    Returns:
        None
    """
    template = os.path.join(inputdir, "baci_local_shape_lm_template.json")
    input_file = os.path.join(experiment_directory, "baci_local_shape_lm.json")
    third_party_input_file = os.path.join(
        third_party_inputs, "baci_input_files", "lm_tri_fsi_shape_template.dat"
    )
    third_party_input_file_monitor = os.path.join(
        third_party_inputs, "baci_input_files", "lm_tri_fsi_shape_E2000_nue03_p.monitor"
    )
    experiment_name = "OptmizeBaciLM_" + json.dumps(singularity_bool)

    baci_release, _, _, post_processor = baci_link_paths

    dir_dict = {
        'experiment_name': experiment_name,
        'experiment_dir': str(experiment_directory),
        'baci_input': third_party_input_file,
        'baci_input_monitor': third_party_input_file_monitor,
        'baci-release': baci_release,
        'post_processor': post_processor,
        'singularity_boolean': json.dumps(singularity_bool),
    }

    injector.inject(dir_dict, template, input_file)
    arguments = ['--input=' + input_file, '--output=' + str(experiment_directory)]

    if singularity_bool is True:
        # check existence of local singularity image
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = '../../../../singularity_image.sif'
        abs_path = os.path.join(script_dir, rel_path)
        assert os.path.isfile(abs_path), (
            "No singularity image existent! Please provide an up-to-date "
            "singularity image for the testing framework. Being in the "
            "directory `pqueens` you can run the command:\n"
            "----------------------------------------------------\n"
            "sudo /usr/bin/singularity build ../singularity_image.sif "
            "../singularity_recipe.def\n"
            "----------------------------------------------------\n"
            "to build a fresh image."
        )

        # check if hash of singularity image is correct
        # local hash key
        hashlist = hash_files()
        local_hash = ''.join(hashlist)

        # check singularity image hash
        command_list = ['/usr/bin/singularity', 'run', abs_path, '--hash=true']
        command_string = ' '.join(command_list)
        _, _, singularity_hash_list, _ = run_subprocess(
            command_string, additional_error_message="Singularity hash-check failed!"
        )
        singularity_hash = [
            ele.replace("\'", "") for ele in singularity_hash_list.strip('][').split(', ')
        ]
        singularity_hash = [ele.replace("]", "") for ele in singularity_hash]
        singularity_hash = [ele.replace("\n", "") for ele in singularity_hash]
        singularity_hash = ''.join(singularity_hash)

        assert local_hash == singularity_hash, (
            "Local hash key and singularity image hash key "
            "deviate! Please provide and up-to-date "
            "singularity image by running:\n"
            "----------------------------------------------------\n"
            "sudo /usr/bin/singularity build ../singularity_image.sif "
            "../singularity_recipe.def\n"
            "----------------------------------------------------\n"
        )

    main(arguments)

    result_file_name = experiment_name + ".csv"
    result_file = os.path.join(experiment_directory, result_file_name)

    result_data = pd.read_csv(
        result_file,
        sep='\t',
    )

    np.testing.assert_equal(result_data['iter'][1], 1)
    np.testing.assert_allclose(result_data['resnorm'][1], 1.42069484e-02, 1.0e-5)
    np.testing.assert_allclose(result_data['gradnorm'][1], 4.53755307e-03, 1.0e-5)

    params = result_data["params"].str.replace(r"[", "")
    params = params.str.replace(r"]", "")
    params = params.str.split(expand=True)
    np.testing.assert_allclose(params.loc[1].astype(float), [1.70123614e03, 3.43936558e-01], 1.0e-5)

    delta_params = result_data["delta_params"].str.replace(r"[", "")
    delta_params = delta_params.str.replace(r"]", "")
    delta_params = delta_params.str.split(expand=True)
    np.testing.assert_allclose(
        delta_params.loc[1].astype(float), [1.28123570e02, -3.47241922e-02], 1.0e-5
    )

    np.testing.assert_allclose(result_data['mu'][1], 1.0, 1.0e-5)
