"""Test suite for integration tests with the cluster.

Elementary Effects simulations with BACI using the INVAAA minimal model.
"""

import pathlib
import pickle

import numpy as np
import pytest

from pqueens.main import main
from pqueens.utils import injector
from pqueens.utils.run_subprocess import run_subprocess


@pytest.mark.lnm_cluster
def test_cluster_elementary_effects(
    inputdir, tmpdir, third_party_inputs, cluster_testsuite_settings, baci_cluster_paths
):
    """Test for the Elementary Effects Iterator on the clusters with BACI.

    Args:
        inputdir (str): Path to the JSON input file
        tmpdir (str): Temporary directory in which the pytests are run
        third_party_inputs (str): Path to the BACI input files
        cluster_testsuite_settings (dict): Collection of cluster specific settings

    Returns:
        None
    """
    # unpack cluster settings needed for all cluster tests
    cluster = cluster_testsuite_settings["cluster"]
    cluster_user = cluster_testsuite_settings["cluster_user"]
    cluster_address = cluster_testsuite_settings["cluster_address"]
    cluster_bind = cluster_testsuite_settings["cluster_bind"]
    connect_to_resource = cluster_testsuite_settings["connect_to_resource"]
    cluster_queens_testing_folder = cluster_testsuite_settings["cluster_queens_testing_folder"]
    cluster_path_to_singularity = cluster_testsuite_settings["cluster_path_to_singularity"]
    scheduler_type = cluster_testsuite_settings["scheduler_type"]
    singularity_remote_ip = cluster_testsuite_settings["singularity_remote_ip"]

    path_to_executable = baci_cluster_paths["path_to_executable"]
    path_to_drt_monitor = baci_cluster_paths["path_to_drt_monitor"]
    path_to_drt_ensight = baci_cluster_paths["path_to_drt_ensight"]
    path_to_drt_monitor = baci_cluster_paths["path_to_drt_monitor"]
    path_to_post_processor = baci_cluster_paths["path_to_post_processor"]

    # unique experiment name
    experiment_name = cluster + "_morris_salib"

    template = pathlib.Path(inputdir, "elementary_effects_baci_cluster_invaaa_template.json")
    input_file = pathlib.Path(tmpdir, f"morris_{cluster}_deep_invaaa.json")

    # specific folder for this test
    cluster_experiment_dir = cluster_queens_testing_folder.joinpath(experiment_name)

    baci_input_filename = "invaaa_ee.dat"
    third_party_input_file_local = pathlib.Path(
        third_party_inputs, "baci_input_files", baci_input_filename
    )
    path_to_input_file_cluster = cluster_experiment_dir.joinpath("input")
    input_file_cluster = path_to_input_file_cluster.joinpath(baci_input_filename)

    experiment_dir = cluster_experiment_dir.joinpath("output")

    command_string = f'mkdir -v -p {path_to_input_file_cluster}'
    _, _, stdout, _ = run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_user=cluster_user,
        remote_address=cluster_address,
    )
    print(stdout)

    command_string = f'mkdir -v -p {experiment_dir}'
    _, _, stdout, _ = run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_user=cluster_user,
        remote_address=cluster_address,
    )
    print(stdout)

    # copy input file to cluster
    command = ' '.join(
        [
            'scp',
            str(third_party_input_file_local),
            connect_to_resource + ':' + str(input_file_cluster),
        ]
    )
    _, _, stdout, _ = run_subprocess(command)
    print(stdout)

    dir_dict = {
        'experiment_name': str(experiment_name),
        'path_to_singularity': str(cluster_path_to_singularity),
        'input_template': str(input_file_cluster),
        'path_to_executable': str(path_to_executable),
        'path_to_drt_monitor': str(path_to_drt_monitor),
        'path_to_drt_ensight': str(path_to_drt_ensight),
        'path_to_post_processor': str(path_to_post_processor),
        'experiment_dir': str(experiment_dir),
        'connect_to_resource': connect_to_resource,
        'cluster_bind': cluster_bind,
        'cluster': cluster,
        'scheduler_type': scheduler_type,
        'singularity_remote_ip': singularity_remote_ip,
    }

    injector.inject(dir_dict, template, input_file)
    arguments = ['--input=' + str(input_file), '--output=' + str(tmpdir)]

    main(arguments)

    result_file = pathlib.Path(tmpdir, experiment_name + '.pickle')
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # test results of SA analysis
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu"], np.array([-1.361395, 0.836351]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star"], np.array([1.361395, 0.836351]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["sigma"], np.array([0.198629, 0.198629]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star_conf"], np.array([0.136631, 0.140794]), rtol=1.0e-3
    )
