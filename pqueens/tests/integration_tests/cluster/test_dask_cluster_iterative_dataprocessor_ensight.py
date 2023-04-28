"""Test remote BACI simulations with ensight data-processor."""
import logging
import pathlib
import pickle

import mock
import numpy as np
import pandas as pd
import pytest

import pqueens.database.database as DB_module
import pqueens.parameters.parameters as parameters_module
from conftest import bruteforce_cluster_settings, charon_cluster_settings, deep_cluster_settings
from pqueens.data_processor.data_processor_ensight import DataProcessorEnsight
from pqueens.iterators import from_config_create_iterator
from pqueens.main import get_config_dict
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
def test_cluster_baci_data_processor_ensight(
    inputdir,
    tmpdir,
    third_party_inputs,
    monkeypatch,
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
        tmpdir (Path): Temporary directory in which the pytests are run
        third_party_inputs (str): Path to the BACI input files
        monkeypatch: fixture for monkey-patching
        dask_cluster_settings (dict): Cluster settings
        cluster_user (str): Cluster user
        remote_queens_repository (str): Path to queens repository on remote host
        remote_python (str): Path to Python environment on remote host
    """
    # monkeypatch the "input" function, so that it returns "y".
    # This simulates the user entering "y" in the terminal:
    monkeypatch.setattr('builtins.input', lambda _: "y")

    base_directory = pathlib.Path("$HOME", "workspace", "build")

    path_to_executable = base_directory / "baci-release"
    path_to_drt_ensight = base_directory / "post_drt_ensight"

    # unique experiment name
    pytest_name = pathlib.Path(tmpdir).parents[0].stem
    experiment_name = f"test_{dask_cluster_settings['name']}_data_processor_ensight"

    def patch_experiments_directory(_):
        """Base directory for all experiments on the computing machine."""
        experiments_dir = pathlib.Path.home() / 'queens-testing' / pytest_name / experiment_name
        pathlib.Path.mkdir(experiments_dir, parents=True, exist_ok=True)
        return experiments_dir

    config_directories_dask.experiment_directory = patch_experiments_directory

    # specific folder for this test
    baci_input_template_name = "invaaa_ee.dat"
    baci_input_file_template = pathlib.Path(
        third_party_inputs, "baci_input_files", baci_input_template_name
    )

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

    with DB_module.database:  # pylint: disable=no-member

        parameters_module.from_config_create_parameters(config)

        def patch_data(*_args):
            """Patch reading experimental data from database."""
            return (
                experiment_name,
                pd.DataFrame.from_dict({"x1": [-16, 10], "x2": [7, 15], "x3": [0.63, 0.2]}),
                ['x1', 'x2', 'x3'],
                None,
            )

        # create iterator
        with mock.patch.object(DataProcessorEnsight, '_get_experimental_data_from_db', patch_data):
            iterator = from_config_create_iterator(config)

        # Create a BACI model for the benchmarks
        model = iterator.model

        # Evaluate the first batch
        first_sample_batch = np.array([[0.2, 10], [0.3, 20], [0.45, 100]])
        first_batch = np.array(model.evaluate(first_sample_batch)["mean"])

        # Evaluate a second batch
        # In order to make sure that no port is closed after one batch
        second_sample_batch = np.array([[0.25, 25], [0.4, 46], [0.47, 211]])
        second_batch = np.array(model.evaluate(second_sample_batch)["mean"])

        # Third batch with MC samples
        iterator.run()

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

    result_file_name = experiment_name + ".pickle"

    result_file = tmpdir / result_file_name
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    reference_mc_mean = np.array([-1.39371249, 1.72861153])
    reference_mc_var = np.array([0.01399851, 0.02759816])
    np.testing.assert_array_almost_equal(reference_mc_mean, results['mean'])
    np.testing.assert_array_almost_equal(reference_mc_var, results['var'])
