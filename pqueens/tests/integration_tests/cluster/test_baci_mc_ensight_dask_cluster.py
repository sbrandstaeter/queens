"""Test remote BACI simulations with ensight data-processor."""
import logging
from pathlib import Path

import numpy as np
import pytest

from pqueens.main import run
from pqueens.tests.integration_tests.conftest import (  # BRUTEFORCE_CLUSTER_TYPE,
    CHARON_CLUSTER_TYPE,
    THOUGHT_CLUSTER_TYPE,
)
from pqueens.utils import config_directories, injector, io_utils

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
    def mock_experiment_dir(
        self,
        tmp_path_factory,
        monkeypatch,
        connect_to_resource,
    ):
        """Mock the experiment directory on the cluster.

        The goal is to separate the testing data from production data of the user.
        NOTE: It is necessary to mock the whole experiment_directory method.
        Otherwise, the mock is not loaded properly remote.
        This is in contrast to the local mocking where it suffices to mock
        config_directories.EXPERIMENTS_BASE_FOLDER_NAME.
        Note that we also rely on this local mock here!
        """
        pytest_basename = tmp_path_factory.getbasetemp().name

        def patch_experiments_directory(experiment_name):
            """Base directory for all experiments on the computing machine."""
            experiments_dir = (
                Path.home()
                / config_directories.BASE_DATA_DIR
                / config_directories.TESTS_BASE_FOLDER_NAME
                / pytest_basename
                / experiment_name
            )
            Path.mkdir(experiments_dir, parents=True, exist_ok=True)
            return experiments_dir

        monkeypatch.setattr(config_directories, "experiment_directory", patch_experiments_directory)
        _logger.debug("Mocking of dask experiment_directory  was successful.")
        _logger.debug(
            "dask experiment_directory is mocked to '$HOME/%s/%s/%s/<experiment_name>' on %s",
            config_directories.BASE_DATA_DIR,
            config_directories.EXPERIMENTS_BASE_FOLDER_NAME,
            pytest_basename,
            connect_to_resource,
        )

    @pytest.fixture(scope="session")
    def remote_python(self, pytestconfig, cluster_settings):
        """Path to Python environment on remote host."""
        remote_python = pytestconfig.getoption("remote_python")
        if remote_python is None:
            remote_python = cluster_settings["default_python_path"]
        return remote_python

    @pytest.fixture(scope="session")
    def remote_queens_repository(self, pytestconfig):
        """Path to queens repository on remote host."""
        return pytestconfig.getoption("remote_queens_repository")

    def test_baci_mc_ensight_cluster(
        self,
        inputdir,
        tmp_path,
        third_party_inputs,
        cluster_settings,
        cluster_user,
        baci_cluster_paths,
        remote_queens_repository,
        remote_python,
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
            cluster_user (str): name or id of the account to log in on cluster
            baci_cluster_paths (dict): collection of paths to BACI executables on the cluster
            remote_queens_repository (str): Path to QUEENS repository on remote host
            remote_python (str): Path to Python environment on remote host
            baci_example_expected_mean (np.ndarray): Expected mean for the MC samples
            baci_example_expected_var (np.ndarray): Expected var for the MC samples
            baci_example_expected_output (np.ndarray): Expected output for the MC samples
        """
        cluster_name = cluster_settings["name"]

        # unique experiment name
        experiment_name = f"baci_mc_ensight_{cluster_name}"

        baci_input_file_template = Path(
            third_party_inputs, "baci_input_files", "meshtying3D_patch_lin_duallagr_new_struct.dat"
        )

        template_options = {
            **baci_cluster_paths,
            **cluster_settings,
            'experiment_name': experiment_name,
            'input_template': baci_input_file_template,
            'cluster_user': cluster_user,
            'cluster_python_path': remote_python,
            'cluster_queens_repository': remote_queens_repository,
        }
        queens_input_file_template = inputdir / "baci_mc_ensight_cluster_template.yml"
        queens_input_file = tmp_path / f"baci_mc_ensight_cluster_{cluster_name}.yml"
        injector.inject(template_options, queens_input_file_template, queens_input_file)

        # get json file as config dictionary
        run(queens_input_file, tmp_path)

        # Check if we got the expected results
        result_file_name = tmp_path / f"{experiment_name}.pickle"
        results = io_utils.load_result(result_file_name)

        # assert statements
        np.testing.assert_array_almost_equal(results['mean'], baci_example_expected_mean, decimal=6)
        np.testing.assert_array_almost_equal(results['var'], baci_example_expected_var, decimal=6)
        np.testing.assert_array_almost_equal(
            results['raw_output_data']['mean'], baci_example_expected_output, decimal=6
        )
