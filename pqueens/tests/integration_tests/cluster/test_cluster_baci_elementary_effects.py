"""Test suite for integration tests with the cluster.

Elementary Effects simulations with BACI using the INVAAA minimal model.
"""
import logging
import pathlib
import pickle
from pathlib import Path

import numpy as np
import pytest

from pqueens import run
from pqueens.schedulers.cluster_scheduler import (
    BRUTEFORCE_CLUSTER_TYPE,
    CHARON_CLUSTER_TYPE,
    DEEP_CLUSTER_TYPE,
)
from pqueens.utils import injector
from pqueens.utils.config_directories import experiment_directory
from pqueens.utils.run_subprocess import run_subprocess

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
def test_cluster_baci_elementary_effects(
    inputdir, tmpdir, third_party_inputs, cluster_testsuite_settings, baci_cluster_paths
):
    """Test for the Elementary Effects Iterator on the clusters with BACI.

    Args:
        inputdir (str): Path to the JSON input file
        tmpdir (str): Temporary directory in which the pytests are run
        third_party_inputs (str): Path to the BACI input files
        cluster_testsuite_settings (dict): Collection of cluster specific settings

    Returns:
        None
    """
    # unpack cluster settings needed for all cluster tests
    cluster = cluster_testsuite_settings["cluster"]
    connect_to_resource = cluster_testsuite_settings["connect_to_resource"]
    singularity_remote_ip = cluster_testsuite_settings["singularity_remote_ip"]

    path_to_executable = baci_cluster_paths["path_to_executable"]
    path_to_drt_ensight = baci_cluster_paths["path_to_drt_ensight"]
    path_to_drt_monitor = baci_cluster_paths["path_to_drt_monitor"]
    path_to_post_processor = baci_cluster_paths["path_to_post_processor"]

    # unique experiment name
    experiment_name = f"test_{cluster}_morris_salib"

    # specific folder for this test
    baci_input_template_name = "invaaa_ee.dat"
    local_baci_input_file_template = pathlib.Path(
        third_party_inputs, "baci_input_files", baci_input_template_name
    )
    cluster_experiment_dir = experiment_directory(
        experiment_name, remote_connect=connect_to_resource
    )
    cluster_baci_input_file_template_dir = cluster_experiment_dir.joinpath("input")
    cluster_baci_input_file_template = cluster_baci_input_file_template_dir.joinpath(
        baci_input_template_name
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
    template_options = {
        'experiment_name': str(experiment_name),
        'input_template': str(cluster_baci_input_file_template),
        'path_to_executable': str(path_to_executable),
        'path_to_drt_monitor': str(path_to_drt_monitor),
        'path_to_drt_ensight': str(path_to_drt_ensight),
        'path_to_post_processor': str(path_to_post_processor),
        'connect_to_resource': connect_to_resource,
        'cluster': cluster,
        'singularity_remote_ip': singularity_remote_ip,
    }
    queens_input_file_template = pathlib.Path(
        inputdir, "baci_cluster_elementary_effects_template.json"
    )
    queens_input_file = pathlib.Path(tmpdir, f"elementary_effects_{cluster}_deep_invaaa.json")
    injector.inject(template_options, queens_input_file_template, queens_input_file)

    run(Path(queens_input_file), Path(tmpdir))

    result_file = pathlib.Path(tmpdir, experiment_name + '.pickle')
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # test results of SA analysis
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu"], np.array([-1.361395, 0.836351]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star"], np.array([1.361395, 0.836351]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["sigma"], np.array([0.198629, 0.198629]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star_conf"], np.array([0.136631, 0.140794]), rtol=1.0e-3
    )
