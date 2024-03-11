"""Test remote BACI simulations with ensight data-processor."""

import logging
from pathlib import Path

import numpy as np
import pytest

import queens.schedulers.cluster_scheduler as cluster_scheduler  # pylint: disable=consider-using-from-import
from queens.main import run
from queens.utils import config_directories, injector, io_utils
from queens.utils.fcc_utils import from_config_create_object
from tests.integration_tests.conftest import (  # BRUTEFORCE_CLUSTER_TYPE,
    CHARON_CLUSTER_TYPE,
    THOUGHT_CLUSTER_TYPE,
)

_logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "cluster",
    [
        pytest.param(THOUGHT_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster),
        # pytest.param(BRUTEFORCE_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster),
        pytest.param(CHARON_CLUSTER_TYPE, marks=pytest.mark.imcs_cluster),
    ],
    indirect=True,
)
class TestDaskCluster:
    """Test class collecting all test with Dask jobqueue clusters and Baci.

    NOTE: we use a class here since our fixture are set to autouse, but we only want to call them
    for these tests.
    """

    @pytest.fixture(autouse=True)
    def mock_experiment_dir(self, monkeypatch, cluster_settings, patched_base_directory):
        """Mock the experiment directory on the cluster.

        The goal is to separate the testing data from production data of the user.
        NOTE: It is necessary to mock the whole experiment_directory method.
        Otherwise, the mock is not loaded properly remote.
        This is in contrast to the local mocking where it suffices to mock
        config_directories.EXPERIMENTS_BASE_FOLDER_NAME.
        Note that we also rely on this local mock here!
        """

        def patch_experiments_directory(experiment_name):
            """Base directory for all experiments on the computing machine."""
            experiments_dir = (
                Path(patched_base_directory.replace("$HOME", str(Path.home()))) / experiment_name
            )
            Path.mkdir(experiments_dir, parents=True, exist_ok=True)
            return experiments_dir

        monkeypatch.setattr(cluster_scheduler, "experiment_directory", patch_experiments_directory)
        _logger.debug("Mocking of dask experiment_directory  was successful.")
        _logger.debug(
            "dask experiment_directory is mocked to '%s/<experiment_name>' on %s@%s",
            patched_base_directory,
            cluster_settings["user"],
            cluster_settings["host"],
        )

    def experiment_dir_on_cluster(self):
        """Remote experiment path."""
        return (
            f"$HOME/{config_directories.BASE_DATA_DIR}/{config_directories.TESTS_BASE_FOLDER_NAME}"
        )

    @pytest.fixture(name="patched_base_directory")
    def fixture_patched_base_directory(self, pytest_id):
        """Path of the tests on the remote cluster."""
        return self.experiment_dir_on_cluster() + f"/{pytest_id}"

    def test_baci_mc_ensight_cluster(
        self,
        inputdir,
        tmp_path,
        third_party_inputs,
        cluster_settings,
        baci_cluster_paths,
        baci_example_expected_mean,
        baci_example_expected_var,
        baci_example_expected_output,
    ):
        """Test remote BACI simulations with DASK jobqueue and MC iterator.

        Test for remote BACI simulations on a remote cluster in combination
        with
        - DASK jobqueue cluster
        - Monte-Carlo (MC) iterator
        - BACI ensight data-processor.


        Args:
            self:
            inputdir (Path): Path to the JSON input file
            tmp_path (Path): Temporary directory for this test
            third_party_inputs (str): Path to the BACI input files
            cluster_settings (dict): Cluster settings
            baci_cluster_paths (dict): collection of paths to BACI executables on the cluster
            baci_example_expected_mean (np.ndarray): Expected mean for the MC samples
            baci_example_expected_var (np.ndarray): Expected var for the MC samples
            baci_example_expected_output (np.ndarray): Expected output for the MC samples
            patched_base_directory (str): directory of the test simulation data on the cluster
        """
        cluster_name = cluster_settings["name"]

        # unique experiment name
        experiment_name = f"baci_mc_ensight_{cluster_name}"

        baci_input_file_template = Path(
            third_party_inputs, "baci", "meshtying3D_patch_lin_duallagr_new_struct.dat"
        )

        template_options = {
            **baci_cluster_paths,
            **cluster_settings,
            'experiment_name': experiment_name,
            'input_template': baci_input_file_template,
        }
        queens_input_file_template = inputdir / "baci_mc_ensight_cluster_template.yml"
        queens_input_file = tmp_path / f"baci_mc_ensight_cluster_{cluster_name}.yml"
        injector.inject(
            template_options, queens_input_file_template, queens_input_file, strict=False
        )

        # get json file as config dictionary
        run(queens_input_file, tmp_path)

        # The data has to be deleted before the assertion
        self.delete_simulation_data(queens_input_file)

        # Check if we got the expected results
        result_file_name = tmp_path / f"{experiment_name}.pickle"
        results = io_utils.load_result(result_file_name)

        # assert statements
        np.testing.assert_array_almost_equal(results['mean'], baci_example_expected_mean, decimal=6)
        np.testing.assert_array_almost_equal(results['var'], baci_example_expected_var, decimal=6)
        np.testing.assert_array_almost_equal(
            results['raw_output_data']['result'], baci_example_expected_output, decimal=6
        )

    def delete_simulation_data(self, input_file_path):
        """Delete simulation data on the cluster.

        This approach deletes test simulation data older then 7 days
        Args:
            input_file_path (pathlib.Path): Path to input file
        """
        # Create a remote connection
        remote_connection_option = io_utils.load_input_file(input_file_path)["my_remote_connection"]
        remote_connection = from_config_create_object(remote_connection_option)
        remote_connection.open()

        # Delete data from tests older then 1 week
        command = (
            "find "
            + str(self.experiment_dir_on_cluster())
            + " -mtime +7 -type d -exec rm -rv {} \\;"
        )
        result = remote_connection.run(command, in_stream=False)
        _logger.debug("Deleting old simulation data:\n%s", result.stdout)
