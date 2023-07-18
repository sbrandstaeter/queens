"""Test suite for integration tests with the cluster and singularity."""

import logging
from pathlib import Path

import numpy as np
import pytest

from pqueens import run
from pqueens.schedulers.cluster_scheduler import (
    BRUTEFORCE_CLUSTER_TYPE,
    CHARON_CLUSTER_TYPE,
    DEEP_CLUSTER_TYPE,
)
from pqueens.utils import config_directories, injector
from pqueens.utils.config_directories import experiment_directory
from pqueens.utils.io_utils import load_result
from pqueens.utils.manage_singularity import SingularityManager
from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "cluster",
    [
        # pytest.param(DEEP_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster),
        pytest.param(BRUTEFORCE_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster),
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

    def test_baci_mc_ensight_singularity_cluster(
        self,
        inputdir,
        tmp_path,
        cluster_settings,
        prepare_baci_input_template_remote,
        baci_cluster_paths,
        baci_example_expected_mean,
        baci_example_expected_var,
    ):
        """Test BACI with MC using singularity.

        Args:
            inputdir (Path): Path to the JSON input file
            tmp_path (Path): Temporary directory for this test
            cluster_settings (dict): Collection of cluster specific settings
            prepare_baci_input_template_remote (fct): helper function to copy template to remote
            baci_cluster_paths (dict): collection of paths to BACI executables on the cluster
            baci_elementary_effects_check_results (function): function to check the results
            baci_example_expected_mean (np.ndarray): Expected mean for the MC samples
            baci_example_expected_var (np.ndarray): Expected var for the MC samples
        """
        cluster_name = cluster_settings["name"]

        # unique experiment name
        experiment_name = f"baci_mc_ensight_singularity_{cluster_name}"

        baci_input_template_name = "meshtying3D_patch_lin_duallagr_new_struct.dat"
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

        queens_input_file_template = Path(
            inputdir, "baci_mc_ensight_singularity_cluster_template.yml"
        )
        queens_input_file = tmp_path / f"baci_mc_ensight_singularity_cluster_{cluster_name}.yml"
        injector.inject(template_options, queens_input_file_template, queens_input_file)

        run(queens_input_file, tmp_path)

        result_file = tmp_path / f"{experiment_name}.pickle"

        results = load_result(result_file)

        # assert statements
        np.testing.assert_array_almost_equal(results['mean'], baci_example_expected_mean, decimal=6)
        np.testing.assert_array_almost_equal(results['var'], baci_example_expected_var, decimal=6)
