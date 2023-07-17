"""Test suite for native cluster tests."""
import logging

import numpy as np
import pytest

from pqueens import run
from pqueens.schedulers.cluster_scheduler import BRUTEFORCE_CLUSTER_TYPE, DEEP_CLUSTER_TYPE
from pqueens.utils import injector
from pqueens.utils.io_utils import load_result

_logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "cluster",
    [
        pytest.param(DEEP_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster_deep_native),
        pytest.param(BRUTEFORCE_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster_bruteforce_native),
    ],
    indirect=True,
)
def test_baci_mc_ensight_native_cluster(
    inputdir,
    tmp_path,
    third_party_inputs,
    baci_cluster_paths_native,
    cluster,
    baci_example_expected_mean,
    baci_example_expected_var,
):
    """Test for the Elementary Effects Iterator on the clusters with BACI.

    Args:
        inputdir (Path): Path to the JSON input file
        tmp_path (Path): Temporary directory in which the pytests are run
        third_party_inputs (Path): Path to the BACI input files
        baci_cluster_paths_native (dict): Paths to baci native on cluster
        cluster (str): Cluster name
        baci_example_expected_mean (np.ndarray): Expected mean for the MC samples
        baci_example_expected_var (np.ndarray): Expected var for the MC samples
    """
    path_to_executable = baci_cluster_paths_native["path_to_executable"]
    path_to_post_ensight = baci_cluster_paths_native["path_to_post_ensight"]

    experiment_name = f"baci_mc_ensight_native_{cluster}"

    template = inputdir / "baci_mc_ensight_native_cluster_template.yml"
    input_file = tmp_path / f"baci_mc_ensight_native_cluster_{cluster}.yml"

    baci_input_filename = "meshtying3D_patch_lin_duallagr_new_struct.dat"
    third_party_input_file_local = third_party_inputs / "baci_input_files" / baci_input_filename

    dir_dict = {
        'experiment_name': experiment_name,
        'input_template': third_party_input_file_local,
        'path_to_executable': path_to_executable,
        'path_to_drt_monitor': path_to_post_ensight,
        'cluster': cluster,
    }

    injector.inject(dir_dict, template, input_file)
    run(input_file, tmp_path)

    result_file = tmp_path / f"{experiment_name}.pickle"
    results = load_result(result_file)

    # assert statements
    np.testing.assert_array_almost_equal(results['mean'], baci_example_expected_mean, decimal=6)
    np.testing.assert_array_almost_equal(results['var'], baci_example_expected_var, decimal=6)
