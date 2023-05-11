"""Test remote BACI simulations with ensight data-processor."""
import logging
import pickle
from pathlib import Path

import mock
import numpy as np
import pandas as pd
import pytest

from pqueens.data_processor.data_processor_ensight import DataProcessorEnsight
from pqueens.main import run
from pqueens.tests.integration_tests.cluster.conftest import (
    bruteforce_cluster_settings,
    charon_cluster_settings,
    deep_cluster_settings,
)
from pqueens.utils import config_directories, injector

_logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "dask_cluster_settings",
    [
        pytest.param(deep_cluster_settings, marks=pytest.mark.lnm_cluster, id="deep"),
        pytest.param(bruteforce_cluster_settings, marks=pytest.mark.lnm_cluster, id="bruteforce"),
        pytest.param(charon_cluster_settings, marks=pytest.mark.imcs_cluster, id="charon"),
    ],
)
class TestDaskCluster:
    """Test class collecting all test with Dask jobqueue clusters and Baci."""

    @pytest.fixture(autouse=True)
    def mock_experiment_dir(
        self,
        cluster_user,
        dask_cluster_settings,
        tmp_path_factory,
        monkeypatch,
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
            "dask experiment_directory is mocked to '$HOME/%s/%s/%s/<experiment_name>' on %s@%s",
            config_directories.BASE_DATA_DIR,
            config_directories.EXPERIMENTS_BASE_FOLDER_NAME,
            pytest_basename,
            cluster_user,
            dask_cluster_settings["cluster_address"],
        )

    def test_cluster_baci_mc(
        self,
        inputdir,
        tmp_path,
        third_party_inputs,
        dask_cluster_settings,
        cluster_user,
        remote_queens_repository,
        remote_python,
    ):
        """Test remote BACI simulations with DASK jobqueue and MC iterator.

        Test for remote BACI simulations on a remote cluster in combination
        with
        - DASK jobqueue cluster
        - Monte-Carlo (MC) iterator
        - BACI ensight data-processor.
        The test is based on the MC iterator.


        Args:
            self:
            inputdir (Path): Path to the JSON input file
            tmp_path (Path): Temporary directory in which the pytests are run
            third_party_inputs (str): Path to the BACI input files
            dask_cluster_settings (dict): Cluster settings
            cluster_user (str): Cluster user
            remote_queens_repository (str): Path to queens repository on remote host
            remote_python (str): Path to Python environment on remote host
        """
        base_directory = Path("$HOME", "workspace", "build")

        path_to_executable = base_directory / "baci-release"
        path_to_drt_ensight = base_directory / "post_drt_ensight"

        # unique experiment name
        experiment_name = f"test_{dask_cluster_settings['name']}_data_processor_ensight"

        baci_input_file_template = Path(third_party_inputs, "baci_input_files", "invaaa_ee.dat")

        if remote_python is None:
            remote_python = dask_cluster_settings['cluster_python_path']
        if remote_queens_repository is None:
            remote_queens_repository = 'null'

        template_options = {
            'experiment_name': str(experiment_name),
            'workload_manager': dask_cluster_settings['workload_manager'],
            'cluster_address': dask_cluster_settings['cluster_address'],
            'cluster_internal_address': dask_cluster_settings['cluster_internal_address'],
            'cluster_user': cluster_user,
            'cluster_python_path': remote_python,
            'cluster_queens_repository': remote_queens_repository,
            'path_to_jobscript': dask_cluster_settings['path_to_jobscript'],
            'cluster_script_path': dask_cluster_settings['cluster_script_path'],
            'input_template': str(baci_input_file_template),
            'path_to_executable': str(path_to_executable),
            'path_to_drt_ensight': str(path_to_drt_ensight),
        }
        queens_input_file_template = inputdir / "baci_dask_cluster_data_processor_ensight.yml"
        queens_input_file = (
            tmp_path / f"baci_cluster_data_processor_ensight_{dask_cluster_settings['name']}.yml"
        )
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

        result_file_name = experiment_name + ".pickle"

        result_file = tmp_path / result_file_name
        with open(result_file, 'rb') as handle:
            results = pickle.load(handle)

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
        np.testing.assert_array_almost_equal(reference_mc_mean, results['mean'])
        np.testing.assert_array_almost_equal(reference_mc_var, results['var'])
        np.testing.assert_array_almost_equal(reference_output, results['raw_output_data']['mean'])
