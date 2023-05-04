"""Test remote BACI simulations with ensight data-processor."""
import logging
import pickle
from pathlib import Path

import mock
import numpy as np
import pandas as pd
import pytest

from conftest import bruteforce_cluster_settings, charon_cluster_settings, deep_cluster_settings
from pqueens.data_processor.data_processor_ensight import DataProcessorEnsight
from pqueens.main import run
from pqueens.utils import config_directories_dask, injector

_logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "dask_cluster_settings",
    [
        pytest.param(deep_cluster_settings, marks=pytest.mark.lnm_cluster),
        pytest.param(bruteforce_cluster_settings, marks=pytest.mark.lnm_cluster),
        pytest.param(charon_cluster_settings, marks=pytest.mark.imcs_cluster),
    ],
)
def test_cluster_baci_mc(
    inputdir,
    tmp_path,
    third_party_inputs,
    dask_cluster_settings,
    cluster_user,
    remote_queens_repository,
    remote_python,
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
    pytest_name = Path(tmp_path).parents[0].stem
    experiment_name = f"test_{dask_cluster_settings['name']}_data_processor_ensight"

    def patch_experiments_directory(_):
        """Base directory for all experiments on the computing machine."""
        experiments_dir = Path.home() / 'queens-testing' / pytest_name / experiment_name
        Path.mkdir(experiments_dir, parents=True, exist_ok=True)
        return experiments_dir

    config_directories_dask.experiment_directory = patch_experiments_directory

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
