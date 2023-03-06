"""Test suite for integration tests with the cluster.

Elementary Effects simulations with BACI using the INVAAA minimal model.
"""
import logging
import pathlib

import pytest

from pqueens import run
from pqueens.schedulers.cluster_scheduler import BRUTEFORCE_CLUSTER_TYPE, CHARON_CLUSTER_TYPE
from pqueens.utils import injector
from pqueens.utils.config_directories import experiment_directory
from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


def test_cluster_baci_data_processor_ensight(
    inputdir,
    tmpdir,
    third_party_inputs,
    dask_cluster_settings,
    baci_elementary_effects_check_results,
):
    """Test for the Elementary Effects Iterator on the clusters with BACI.

    Args:
        inputdir (str): Path to the JSON input file
        tmpdir (str): Temporary directory in which the pytests are run
        third_party_inputs (str): Path to the BACI input files
        dask_cluster_settings (dict): Cluster settings
    """

    base_directory = pathlib.Path("$HOME", "workspace", "build")

    path_to_executable = base_directory / "baci-release"
    path_to_drt_monitor = base_directory / "post_drt_monitor"
    path_to_post_processor = base_directory / "post_processor"
    path_to_drt_ensight = base_directory / "post_drt_ensight"

    # unique experiment name
    experiment_name = f"test_dask_{dask_cluster_settings['name']}_elementary_effects"

    # specific folder for this test
    baci_input_template_name = "invaaa_ee.dat"
    baci_input_file_template = pathlib.Path(
        third_party_inputs, "baci_input_files", baci_input_template_name
    )

    template_options = {
        'experiment_name': str(experiment_name),
        'workload_manager': dask_cluster_settings['workload_manager'],
        'cluster_address': dask_cluster_settings['cluster_address'],
        'cluster_python_path': dask_cluster_settings['cluster_python_path'],
        'path_to_jobscript': dask_cluster_settings['path_to_jobscript'],
        'input_template': str(baci_input_file_template),
        'path_to_executable': str(path_to_executable),
        'path_to_drt_monitor': str(path_to_drt_monitor),
        'path_to_drt_ensight': str(path_to_drt_ensight),
        'path_to_post_processor': str(path_to_post_processor),
    }
    queens_input_file_template = pathlib.Path(
        inputdir, "baci_dask_cluster_elementary_effects_template.yml"
    )
    queens_input_file = pathlib.Path(
        tmpdir, f"baci_dask_cluster_elementary_effects_{dask_cluster_settings['name']}.yml"
    )
    injector.inject(template_options, queens_input_file_template, queens_input_file)

    run(queens_input_file, pathlib.Path(tmpdir))

    result_file = pathlib.Path(tmpdir, experiment_name + '.pickle')
    baci_elementary_effects_check_results(result_file)
