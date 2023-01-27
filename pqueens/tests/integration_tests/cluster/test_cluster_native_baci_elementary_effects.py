"""Test suite for native cluster tests.

Elementary Effects simulations with BACI using the INVAAA minimal model.
"""
import logging
from pathlib import Path

import pytest

from pqueens import run
from pqueens.utils import injector

_logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "cluster",
    [
        pytest.param("deep", marks=pytest.mark.lnm_cluster_deep_native),
        pytest.param("bruteforce", marks=pytest.mark.lnm_cluster_bruteforce_native),
    ],
    indirect=True,
)
def test_cluster_native_baci_elementary_effects(
    inputdir,
    tmp_path,
    third_party_inputs,
    baci_cluster_paths_native,
    cluster,
    baci_elementary_effects_check_results,
):
    """Test for the Elementary Effects Iterator on the clusters with BACI.

    Args:
        inputdir (path): Path to the JSON input file
        tmp_path (path): Temporary directory in which the pytests are run
        third_party_inputs (str): Path to the BACI input files
        baci_cluster_paths_native (dict): Paths to baci native on cluster
        cluster (str): Cluster name
        baci_elementary_effects_check_results (function): function to check the results
    """
    path_to_executable = baci_cluster_paths_native["path_to_executable"]
    path_to_drt_monitor = baci_cluster_paths_native["path_to_drt_monitor"]

    experiment_name = cluster + "_native_elementary_effects"

    template = inputdir.joinpath("baci_cluster_native_elementary_effects_template.yml")
    input_file = tmp_path.joinpath(f"elementary_effects_{cluster}_invaaa.yml")

    baci_input_filename = "invaaa_ee.dat"
    third_party_input_file_local = Path(third_party_inputs, "baci_input_files", baci_input_filename)

    dir_dict = {
        'experiment_name': str(experiment_name),
        'input_template': str(third_party_input_file_local),
        'path_to_executable': str(path_to_executable),
        'path_to_drt_monitor': str(path_to_drt_monitor),
        'cluster': cluster,
    }

    injector.inject(dir_dict, template, input_file)
    run(input_file, tmp_path)

    result_file = tmp_path.joinpath(experiment_name + '.pickle')
    baci_elementary_effects_check_results(result_file)
