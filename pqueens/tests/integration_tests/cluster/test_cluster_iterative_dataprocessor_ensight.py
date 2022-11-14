"""Test remote BACI simulations with ensight data-processor."""
import json
import pathlib

import numpy as np
import pytest

import pqueens.database.database as DB_module
import pqueens.parameters.parameters as parameters_module
from pqueens.models import from_config_create_model
from pqueens.utils import injector
from pqueens.utils.run_subprocess import run_subprocess


@pytest.mark.parametrize(
    "cluster",
    [
        pytest.param("deep", marks=pytest.mark.lnm_cluster),
        pytest.param("bruteforce", marks=pytest.mark.lnm_cluster),
        pytest.param("charon", marks=pytest.mark.imcs_cluster),
    ],
    indirect=True,
)
def test_cluster_baci_data_processor_ensight(
    inputdir, tmpdir, third_party_inputs, cluster_testsuite_settings, baci_cluster_paths
):
    """Test remote BACI simulations with ensight data-processor.

    Test suite for remote BACI simulations on the cluster in combination
    with the BACI ensight data-processor. No iterator is used, the model is
    called directly.

    This integration test is constructed such that:
        - The interface-map function is called twice (mimics feedback-loops)
        - The maximum concurrent job is activated
        - data_processor_ensight to remotely communicate with the database (besides the driver)
        - No iterator is used to reduce complexity

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
    path_to_drt_ensight = baci_cluster_paths["path_to_drt_ensight"]
    path_to_drt_monitor = baci_cluster_paths["path_to_drt_monitor"]
    path_to_post_processor = baci_cluster_paths["path_to_post_processor"]

    # unique experiment name
    experiment_name = cluster + "_remote_data_processor_ensight"

    template = pathlib.Path(inputdir, "baci_remote_model_config.json")
    input_file = pathlib.Path(tmpdir, "baci_remote_model_config.json")

    # specific folder for this test
    cluster_experiment_dir = cluster_queens_testing_folder.joinpath(experiment_name)

    baci_input_filename = "invaaa_ee.dat"
    third_party_input_file_local = pathlib.Path(
        third_party_inputs, "baci_input_files", baci_input_filename
    )
    path_to_input_file_cluster = cluster_experiment_dir.joinpath("input")
    baci_input_file_cluster = path_to_input_file_cluster.joinpath(baci_input_filename)

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
            connect_to_resource + ':' + str(baci_input_file_cluster),
        ]
    )
    _, _, stdout, _ = run_subprocess(command)
    print(stdout)

    dir_dict = {
        'experiment_name': str(experiment_name),
        'path_to_singularity': str(cluster_path_to_singularity),
        'input_template': str(baci_input_file_cluster),
        'path_to_executable': str(path_to_executable),
        'path_to_drt_monitor': str(path_to_drt_monitor),
        'path_to_drt_ensight': str(path_to_drt_ensight),
        'path_to_post_processor': str(path_to_post_processor),
        'experiment_dir': str(experiment_dir),
        'connect_to_resource': connect_to_resource,
        'cluster_bind': cluster_bind,
        'cluster': cluster,
        'type': scheduler_type,
        'singularity_remote_ip': singularity_remote_ip,
    }

    injector.inject(dir_dict, template, input_file)

    # Patch the missing config arguments
    with open(str(input_file), encoding="utf8") as f:
        config = json.load(f)
        global_settings = {
            "output_dir": str(tmpdir),
            "experiment_name": config["experiment_name"],
        }
        config["global_settings"] = global_settings
        config["input_file"] = str(input_file)

    # Initialise db module
    DB_module.from_config_create_database(config)

    with DB_module.database as db:  # pylint: disable=no-member

        # Add experimental coordinates to the database
        experimental_data_dict = {"x1": [-16, 10], "x2": [7, 15], "x3": [0.63, 0.2]}
        db.save(experimental_data_dict, experiment_name, 'experimental_data', 1)

        parameters_module.from_config_create_parameters(config)

        # Create a BACI model for the benchmarks
        model = from_config_create_model("model", config)

        # Evaluate the first batch
        first_sample_batch = np.array([[0.2, 10], [0.3, 20], [0.45, 100]])
        first_batch = np.array(model.evaluate(first_sample_batch)["mean"])

        # Evaluate a second batch
        # In order to make sure that no port is closed after one batch
        second_sample_batch = np.array([[0.25, 25], [0.4, 46], [0.47, 211]])
        second_batch = np.array(model.evaluate(second_sample_batch)["mean"][-3:])

    # Check results
    first_batch_reference_solution = np.array(
        [
            [-0.0006949830567464232, 0.0017958658281713724],
            [-0.0012194387381896377, 0.003230389906093478],
            [-0.004366828128695488, 0.0129017299041152],
        ]
    )
    second_batch_reference_solution = np.array(
        [
            [-0.0016464125365018845, 0.004280212335288525],
            [-0.0023093123454600573, 0.00646978011354804],
            [-0.008715332485735416, 0.026327939704060555],
        ]
    )

    np.testing.assert_array_equal(first_batch, first_batch_reference_solution)
    np.testing.assert_array_equal(second_batch, second_batch_reference_solution)
