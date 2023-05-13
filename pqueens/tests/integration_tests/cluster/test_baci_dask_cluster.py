"""Test remote BACI simulations with ensight data-processor."""
import logging
from pathlib import Path

import mock
import numpy as np
import pandas as pd
import pytest

from pqueens.data_processor.data_processor_ensight import DataProcessorEnsight
from pqueens.main import run
from pqueens.schedulers.cluster_scheduler import (
    BRUTEFORCE_CLUSTER_TYPE,
    CHARON_CLUSTER_TYPE,
    DEEP_CLUSTER_TYPE,
)
from pqueens.utils import config_directories, injector, io_utils

_logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "cluster",
    [
        pytest.param(DEEP_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster),
        pytest.param(BRUTEFORCE_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster),
        pytest.param(CHARON_CLUSTER_TYPE, marks=pytest.mark.imcs_cluster),
    ],
    indirect=True,
)
class TestDaskCluster:
    """Test class collecting all test with Dask jobqueue clusters and Baci."""

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
                / config_directories.EXPERIMENTS_BASE_FOLDER_NAME
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

    def test_baci_dask_cluster_monte_carlo(
        self,
        inputdir,
        tmp_path,
        third_party_inputs,
        cluster_settings,
        cluster_user,
        baci_cluster_paths,
        remote_queens_repository,
        remote_python,
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
        """
        cluster_name = cluster_settings["name"]

        # unique experiment name
        experiment_name = f"test_{cluster_name}_monte_carlo"

        baci_input_file_template = Path(third_party_inputs, "baci_input_files", "invaaa_ee.dat")

        template_options = {
            **baci_cluster_paths,
            **cluster_settings,
            'experiment_name': experiment_name,
            'input_template': baci_input_file_template,
            'cluster_user': cluster_user,
            'cluster_python_path': remote_python,
            'cluster_queens_repository': remote_queens_repository,
        }
        queens_input_file_template = inputdir / "baci_dask_cluster_monte_carlo_template.yml"
        queens_input_file = tmp_path / f"baci_dask_cluster_monte_carlo_{cluster_name}.yml"
        injector.inject(template_options, queens_input_file_template, queens_input_file)

        def patch_data(*_args):
            """Patch reading experimental data from database."""
            return (
                experiment_name,
                pd.DataFrame.from_dict({"x1": [-16, 10], "x2": [7, 15], "x3": [0.63, 0.2]}),
                ['x1', 'x2', 'x3'],
                None,
            )

        with mock.patch.object(DataProcessorEnsight, '_get_experimental_data_from_db', patch_data):
            run(queens_input_file, tmp_path)

        result_file = tmp_path / f"{experiment_name}.pickle"
        result = io_utils.load_result(result_file)

        reference_mc_mean = np.array([-1.39371249, 1.72861153])
        reference_mc_var = np.array([0.01399851, 0.02759816])
        reference_output = np.array(
            [
                [-1.40698826, 1.78872097],
                [-1.29981923, 1.56641328],
                [-1.31205046, 1.62528300],
                [-1.55599201, 1.93402886],
            ]
        )
        np.testing.assert_array_almost_equal(reference_mc_mean, result['mean'])
        np.testing.assert_array_almost_equal(reference_mc_var, result['var'])
        np.testing.assert_array_almost_equal(reference_output, result['raw_output_data']['mean'])
