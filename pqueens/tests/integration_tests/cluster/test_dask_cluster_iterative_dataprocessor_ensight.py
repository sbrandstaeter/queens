"""Test remote BACI simulations with ensight data-processor."""
import logging
import pathlib

import numpy as np

import pqueens.database.database as DB_module
import pqueens.parameters.parameters as parameters_module
from pqueens.main import get_config_dict
from pqueens.models import from_config_create_model
from pqueens.utils import injector

_logger = logging.getLogger(__name__)


def test_cluster_baci_data_processor_ensight(
    inputdir, tmpdir, third_party_inputs, monkeypatch, dask_cluster_settings, cluster_user
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
        dask_cluster_settings (dict): Cluster settings
    """
    # monkeypatch the "input" function, so that it returns "y".
    # This simulates the user entering "y" in the terminal:
    monkeypatch.setattr('builtins.input', lambda _: "y")

    base_directory = pathlib.Path("$HOME", "workspace", "build")

    path_to_executable = base_directory / "baci-release"
    path_to_drt_monitor = base_directory / "post_drt_monitor"
    path_to_post_processor = base_directory / "post_processor"
    path_to_drt_ensight = base_directory / "post_drt_ensight"

    # unique experiment name
    experiment_name = f"test_{dask_cluster_settings['name']}_data_processor_ensight"

    # specific folder for this test
    baci_input_template_name = "invaaa_ee.dat"
    baci_input_file_template = pathlib.Path(
        third_party_inputs, "baci_input_files", baci_input_template_name
    )

    template_options = {
        'experiment_name': str(experiment_name),
        'workload_manager': dask_cluster_settings['workload_manager'],
        'cluster_address': dask_cluster_settings['cluster_address'],
        'cluster_user': cluster_user,
        'cluster_python_path': dask_cluster_settings['cluster_python_path'],
        'path_to_jobscript': dask_cluster_settings['path_to_jobscript'],
        'input_template': str(baci_input_file_template),
        'path_to_executable': str(path_to_executable),
        'path_to_drt_monitor': str(path_to_drt_monitor),
        'path_to_drt_ensight': str(path_to_drt_ensight),
        'path_to_post_processor': str(path_to_post_processor),
    }
    queens_input_file_template = pathlib.Path(
        inputdir, "baci_dask_cluster_data_processor_ensight.yml"
    )
    queens_input_file = pathlib.Path(
        tmpdir, f"baci_cluster_data_processor_ensight_{dask_cluster_settings['name']}.yml"
    )
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
        first_batch = np.array(model.evaluate(first_sample_batch)["mean"]).squeeze()

        # Evaluate a second batch
        # In order to make sure that no port is closed after one batch
        second_sample_batch = np.array([[0.25, 25], [0.4, 46], [0.47, 211]])
        second_batch = np.array(model.evaluate(second_sample_batch)["mean"]).squeeze()

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
