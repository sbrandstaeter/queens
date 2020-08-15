"""
Test suite for integration tests for the Morris-Salib Iterator (Elementary Effects) for local
simulations with BACI using the INVAAA minimal model.
"""

import os
import numpy as np
import pickle
from pqueens.utils.manage_singularity import hash_files
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.main import main
from pqueens.utils import injector
import pytest


@pytest.fixture(params=["true", "false"])
def sing_bool(request):
    return request.param


def test_baci_morris_salib(
    inputdir, tmpdir, third_party_inputs, config_dir, set_baci_links_for_gitlab_runner, sing_bool
):
    """
    Integration test for the Salib Morris Iterator together with BACI. The test runs a local native
    BACI simulation as well as a local Singularity based BACI simulation for elementary effects.

    Args:
        inputdir (str): Path to the JSON input file
        tmpdir (str): Temporary directory in which the pytests are run
        third_party_inputs (str): Path to the BACI input files
        config_dir (str): Path to the config directory of QUEENS containing BACI executables
        set_baci_links_for_gitlab_runner (str): Several paths that are needed to build symbolic
                                                links to executables
        sing_bool (str): String that encodes a boolean that is parsed to the JSON input file

    Returns:
        None

    """
    template = os.path.join(inputdir, "morris_baci_local_invaaa_template.json")
    input_file = os.path.join(tmpdir, "morris_baci_local_invaaa.json")
    third_party_input_file = os.path.join(third_party_inputs, "baci_input_files", "invaaa_ee.dat")

    baci_release = os.path.join(config_dir, "baci-release")
    post_drt_monitor = os.path.join(config_dir, "post_drt_monitor")

    # check if symbolic links are existent
    if (not os.path.islink(baci_release)) or (not os.path.islink(post_drt_monitor)):
        # set default baci location for testing machine
        dst_baci, dst_drt_monitor, src_baci, src_drt_monitor = set_baci_links_for_gitlab_runner
        try:
            os.symlink(src_baci, dst_baci)
            os.symlink(src_drt_monitor, dst_drt_monitor)
        except FileNotFoundError:
            raise FileNotFoundError(
                'No working baci-release or post_drt_monitor could be found! '
                'Make sure an appropriate symbolic link is made available '
                'under the config directory! \n'
                'You can create the symbolic links on Linux via:\n'
                '-------------------------------------------------------------------------\n'
                'ln -s <path/to/baci-release> <QUEENS_BaseDir>/config/baci-release\n'
                'ln -s <path/to/post_drt_monitor> <QUEENS_BaseDir>/config/post_drt_monitor\n'
                '-------------------------------------------------------------------------\n'
            )

    dir_dict = {
        'experiment_dir': str(tmpdir),
        'baci_input': third_party_input_file,
        'baci-release': baci_release,
        'post_drt_monitor': post_drt_monitor,
        'singularity_boolean': sing_bool,
    }

    injector.inject(dir_dict, template, input_file)
    arguments = ['--input=' + input_file, '--output=' + str(tmpdir)]

    if sing_bool == 'false':
        # check existence of local singularity image
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = '../../../../driver.simg'
        abs_path = os.path.join(script_dir, rel_path)
        assert os.path.isfile(abs_path), (
            "No singularity image existent! Please provide an up-to-date "
            "singularity image for the testing framework. Being in the "
            "directory `pqueens` you can run the command:\n"
            "----------------------------------------------------\n"
            "sudo /usr/bin/singularity build ../driver.simg "
            "../singularity_recipe\n"
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
        _, _, singularity_hash_list, stderr = run_subprocess(command_string)

        if stderr:
            raise RuntimeError(f'Singularity hash-check return the error: {stderr}. Abort...')
        else:
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
            "sudo /usr/bin/singularity build ../driver.simg "
            "../singularity_recipe\n"
            "----------------------------------------------------\n"
        )

    main(arguments)

    result_file = os.path.join(tmpdir, 'ee_invaaa_local.pickle')
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # test results of SA analysis
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu"], np.array([-1.361395, 0.836351]), rtol=1.0e-3,
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star"], np.array([1.361395, 0.836351]), rtol=1.0e-3,
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["sigma"], np.array([0.198629, 0.198629]), rtol=1.0e-3,
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star_conf"], np.array([0.11853, 0.146817]), rtol=1.0e-3,
    )
