"""Test remote BACI simulations with ensight data-processor."""
import logging
import pathlib

import numpy as np
import pytest

import pqueens.database.database as DB_module
import pqueens.parameters.parameters as parameters_module
from pqueens.main import get_config_dict
from pqueens.models import from_config_create_model
from pqueens.schedulers.cluster_scheduler import (
    BRUTEFORCE_CLUSTER_TYPE,
    CHARON_CLUSTER_TYPE,
    DEEP_CLUSTER_TYPE,
)
from pqueens.utils import injector
from pqueens.utils.config_directories import experiment_directory
from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "cluster",
    [
        # pytest.param(DEEP_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster),
        # pytest.param(BRUTEFORCE_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster),
        pytest.param(CHARON_CLUSTER_TYPE, marks=pytest.mark.imcs_cluster),
    ],
    indirect=True,
)
def test_cluster_baci_data_processor_ensight(
    inputdir,
    tmpdir,
    third_party_inputs,
    cluster_testsuite_settings,
    baci_cluster_paths,
    user,
    monkeypatch,
):
    """Test remote BACI simulations with ensight data-processor.

    Test suite for remote BACI simulations on the cluster in combination
    with the BACI ensight data-processor. No iterator is used, the model is
    called directly.

    This integration test is constructed such that:
        - The interface-map function is called twice (mimics feedback-loops)
        - The maximum concurrent job is activated
        - *data_processor_ensight* to remotely communicate with the database (besides the driver)
        - No iterator is used to reduce complexity

    Args:
        inputdir (str): Path to the JSON input file
        tmpdir (str): Temporary directory in which the pytests are run
        third_party_inputs (str): Path to the BACI input files
        cluster_testsuite_settings (dict): Collection of cluster specific settings
        baci_cluster_paths: TODO_doc
        user (): TODO_doc
    """
    # monkeypatch the "input" function, so that it returns "y".
    # This simulates the user entering "y" in the terminal:
    monkeypatch.setattr('builtins.input', lambda _: "y")

    # unpack cluster settings needed for all cluster tests
    cluster = cluster_testsuite_settings["cluster"]
    connect_to_resource = cluster_testsuite_settings["connect_to_resource"]
    singularity_remote_ip = cluster_testsuite_settings["singularity_remote_ip"]

    path_to_executable = baci_cluster_paths["path_to_executable"]
    path_to_drt_ensight = baci_cluster_paths["path_to_drt_ensight"]
    path_to_drt_monitor = baci_cluster_paths["path_to_drt_monitor"]
    path_to_post_processor = baci_cluster_paths["path_to_post_processor"]

    # unique experiment name
    experiment_name = f"test_{cluster}_data_processor_ensight"

    # specific folder for this test
    baci_input_template_name = "invaaa_ee.dat"
    local_baci_input_file_template = pathlib.Path(
        third_party_inputs, "baci_input_files", baci_input_template_name
    )
    cluster_experiment_dir = experiment_directory(
        experiment_name, remote_connect=connect_to_resource
    )
    cluster_baci_input_file_template_dir = cluster_experiment_dir.joinpath("input")
    cluster_baci_input_file_template = cluster_baci_input_file_template_dir.joinpath(
        baci_input_template_name
    )

    command_string = f'mkdir -v -p {cluster_baci_input_file_template_dir}'
    _, _, stdout, _ = run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_connect=connect_to_resource,
    )
    _logger.info(stdout)

    # copy input file to cluster
    command = ' '.join(
        [
            'scp',
            str(local_baci_input_file_template),
            connect_to_resource + ':' + str(cluster_baci_input_file_template),
        ]
    )
    _, _, stdout, _ = run_subprocess(command)
    _logger.info(stdout)

    template_options = {
        'experiment_name': str(experiment_name),
        'input_template': str(cluster_baci_input_file_template),
        'path_to_executable': str(path_to_executable),
        'path_to_drt_monitor': str(path_to_drt_monitor),
        'path_to_drt_ensight': str(path_to_drt_ensight),
        'path_to_post_processor': str(path_to_post_processor),
        'connect_to_resource': connect_to_resource,
        'cluster': cluster,
        'singularity_remote_ip': singularity_remote_ip,
        'user': user,
    }
    queens_input_file_template = pathlib.Path(inputdir, "baci_cluster_data_processor_ensight.yml")
    queens_input_file = pathlib.Path(tmpdir, "baci_cluster_data_processor_ensight.yml")
    injector.inject(template_options, queens_input_file_template, queens_input_file)

    # Patch the missing config arguments
    config = get_config_dict(queens_input_file, pathlib.Path(tmpdir))

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
