"""Test suite for integration tests with the cluster and singularity."""

import logging
from pathlib import Path

import numpy as np
import pytest

import pqueens.database.database as DB_module
import pqueens.parameters.parameters as parameters_module
from pqueens import run
from pqueens.main import get_config_dict
from pqueens.models import from_config_create_model
from pqueens.schedulers.cluster_scheduler import (
    BRUTEFORCE_CLUSTER_TYPE,
    CHARON_CLUSTER_TYPE,
    DEEP_CLUSTER_TYPE,
)
from pqueens.utils import config_directories, injector
from pqueens.utils.config_directories import experiment_directory
from pqueens.utils.manage_singularity import SingularityManager
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
class TestClusterSingularity:
    """Test class for cluster tests with singularity and BACI."""

    @pytest.fixture(scope="session", autouse=True)
    def prepare_environment_and_singularity(
        self,
        connect_to_resource,
        mock_value_experiments_base_folder_name,
    ):
        """Create clean test environment on cluster and build singularity.

        **WARNING:** Needs to be done AFTER *prepare_cluster_testing_environment*
        to make sure cluster testing folder is clean and existing.
        """
        # Create clean testing environment
        cluster_queens_base_dir = config_directories.remote_base_directory(
            remote_connect=connect_to_resource
        )
        cluster_queens_testing_folder = (
            cluster_queens_base_dir / mock_value_experiments_base_folder_name
        )
        # remove old folder
        _logger.info(
            "Delete testing folder %s on %s", cluster_queens_testing_folder, connect_to_resource
        )
        command_string = f'rm -rfv {cluster_queens_testing_folder}'
        _, _, stdout, _ = run_subprocess(
            command_string=command_string,
            subprocess_type='remote',
            remote_connect=connect_to_resource,
        )
        _logger.info(stdout)

        # create generic testing folder
        _logger.info(
            "Create testing folder %s on %s", cluster_queens_testing_folder, connect_to_resource
        )
        command_string = f'mkdir -v -p {cluster_queens_testing_folder}'
        _, _, stdout, _ = run_subprocess(
            command_string=command_string,
            subprocess_type='remote',
            remote_connect=connect_to_resource,
        )
        _logger.info(stdout)

        # Build singularity
        singularity_manager = SingularityManager(
            singularity_path=cluster_queens_base_dir,
            singularity_bind=None,
            input_file=None,
            remote=True,
            remote_connect=connect_to_resource,
        )
        singularity_manager.prepare_singularity_files()

    @pytest.fixture(autouse=True)
    def mock_input_function(self, monkeypatch):
        """Mock the input function.

        monkeypatch the "input" function, so that it returns "y". This
        simulates the user entering "y" in the terminal:
        """
        monkeypatch.setattr('builtins.input', lambda _: "y")

    @pytest.fixture(scope="session")
    def prepare_baci_input_template_remote(self, third_party_inputs, connect_to_resource):
        """Provide helper function preparing baci input template on remote."""

        def helper_function(baci_input_template_name, experiment_name):
            """Prepare baci input template on remote."""
            local_baci_input_file_template = (
                third_party_inputs / "baci_input_files" / f"{baci_input_template_name}"
            )
            cluster_experiment_dir = experiment_directory(
                experiment_name, remote_connect=connect_to_resource
            )
            cluster_baci_input_file_template_dir = cluster_experiment_dir / "input"
            cluster_baci_input_file_template = (
                cluster_baci_input_file_template_dir / baci_input_template_name
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
            return cluster_baci_input_file_template

        return helper_function

    def test_baci_cluster_elementary_effects(
        self,
        inputdir,
        tmp_path,
        cluster_settings,
        prepare_baci_input_template_remote,
        baci_cluster_paths,
        baci_elementary_effects_check_results,
    ):
        """Test for the Elementary Effects Iterator on the clusters with BACI.

        Using the INVAAA minimal model.

        Args:
            inputdir (Path): Path to the JSON input file
            tmp_path (Path): Temporary directory for this test
            cluster_settings (dict): Collection of cluster specific settings
            prepare_baci_input_template_remote (fct): helper function to copy template to remote
            baci_cluster_paths (dict): collection of paths to BACI executables on the cluster
            baci_elementary_effects_check_results (function): function to check the results
        """
        cluster_name = cluster_settings["name"]

        # unique experiment name
        experiment_name = f"test_{cluster_name}_morris_salib"

        baci_input_template_name = "invaaa_ee.dat"
        cluster_baci_input_file_template = prepare_baci_input_template_remote(
            baci_input_template_name=baci_input_template_name, experiment_name=experiment_name
        )

        template_options = {
            **baci_cluster_paths,
            **cluster_settings,
            'experiment_name': experiment_name,
            'input_template': cluster_baci_input_file_template,
            'cluster': cluster_name,
        }

        queens_input_file_template = Path(inputdir, "baci_cluster_elementary_effects_template.yml")
        queens_input_file = tmp_path / f"baci_cluster_elementary_effects_{cluster_name}.yml"
        injector.inject(template_options, queens_input_file_template, queens_input_file)

        run(queens_input_file, tmp_path)

        result_file = tmp_path / f"{experiment_name}.pickle"
        baci_elementary_effects_check_results(result_file)

    def test_baci_cluster_data_processor_ensight(
        self,
        inputdir,
        tmp_path,
        cluster_settings,
        prepare_baci_input_template_remote,
        baci_cluster_paths,
        user,
    ):
        """Test remote BACI simulations with ensight data-processor.

        Test suite for remote BACI simulations on the cluster in combination
        with the BACI ensight data-processor. No iterator is used, the model is
        called directly.

        This integration test is constructed such that:
            - The interface-map function is called twice (mimics feedback-loops)
            - The maximum concurrent job is activated
            - *data_processor_ensight* communicate with the remote database
            - No iterator is used to reduce complexity

        Args:
            inputdir (Path): Path to the JSON input file
            tmp_path (Path): Temporary directory for this test
            cluster_settings (dict): Collection of cluster specific settings
            prepare_baci_input_template_remote (fct): helper function to copy template to remote
            baci_cluster_paths (dict): collection of paths to BACI executables on the cluster
            user (str): name of the user running the tests
        """
        cluster_name = cluster_settings["name"]

        # unique experiment name
        experiment_name = f"test_{cluster_name}_data_processor_ensight"

        baci_input_template_name = "invaaa_ee.dat"
        cluster_baci_input_file_template = prepare_baci_input_template_remote(
            baci_input_template_name=baci_input_template_name, experiment_name=experiment_name
        )

        template_options = {
            **baci_cluster_paths,
            **cluster_settings,
            'experiment_name': experiment_name,
            'input_template': cluster_baci_input_file_template,
            'cluster': cluster_name,
            'user': user,
        }

        queens_input_file_template = inputdir / "baci_cluster_data_processor_ensight_template.yml"
        queens_input_file = tmp_path / f"baci_cluster_data_processor_ensight_{cluster_name}.yml"
        injector.inject(template_options, queens_input_file_template, queens_input_file)

        # Patch the missing config arguments
        config = get_config_dict(queens_input_file, tmp_path)

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
