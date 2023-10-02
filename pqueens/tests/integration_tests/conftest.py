"""Collect fixtures used by the integration tests."""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from pqueens.utils.path_utils import relative_path_from_queens
from pqueens.utils.run_subprocess import run_subprocess_remote

_logger = logging.getLogger(__name__)

THOUGHT_CLUSTER_TYPE = "thought"
BRUTEFORCE_CLUSTER_TYPE = "bruteforce"
CHARON_CLUSTER_TYPE = "charon"

VALID_PBS_CLUSTER_TYPES = (THOUGHT_CLUSTER_TYPE,)
VALID_SLURM_CLUSTER_TYPES = (BRUTEFORCE_CLUSTER_TYPE, CHARON_CLUSTER_TYPE)

VALID_CLUSTER_CLUSTER_TYPES = VALID_PBS_CLUSTER_TYPES + VALID_SLURM_CLUSTER_TYPES


@dataclass(frozen=True)
class ClusterConfig:
    """Configuration data of cluster.

    Attributes:
        name (str):                         name of cluster
        cluster_address (str):              hostname or address to reach cluster from network
        workload_manager (str):             type of work load scheduling software (PBS or SLURM)
        jobscript_template (Path):          absolute path to jobscript template file
        cluster_internal_address (str)      ip address of login node in cluster internal network
        default_python_path (str):          path indicating the default remote python location
        cluster_script_path (Path):          path to the cluster_script which defines functions
                                            needed for the jobscript
        dask_jobscript_template (Path):     path to the shell script template that runs a
                                            forward solver call (e.g., BACI plus post-processor)
        queue (str, opt):                   Destination queue for each worker job
    """

    name: str
    cluster_address: str
    workload_manager: str
    jobscript_template: Path
    cluster_internal_address: str
    default_python_path: str
    cluster_script_path: Path
    dask_jobscript_template: Path
    queue: Optional[str] = 'null'

    dict = asdict


THOUGHT_CONFIG = ClusterConfig(
    name="thought",
    cluster_address="129.187.58.22",
    workload_manager="slurm",
    queue="normal",
    jobscript_template=relative_path_from_queens("templates/jobscripts/jobscript_thought.sh"),
    cluster_internal_address="null",
    default_python_path="$HOME/anaconda/miniconda/envs/queens/bin/python",
    cluster_script_path=Path("/lnm/share/donottouch.sh"),
    dask_jobscript_template=relative_path_from_queens("templates/jobscripts/jobscript_thought.sh"),
)


BRUTEFORCE_CONFIG = ClusterConfig(
    name="bruteforce",
    cluster_address="bruteforce.lnm.ed.tum.de",
    workload_manager="slurm",
    jobscript_template=relative_path_from_queens("templates/jobscripts/jobscript_bruteforce.sh"),
    cluster_internal_address="10.10.0.1",
    default_python_path="$HOME/anaconda/miniconda/envs/queens/bin/python",
    cluster_script_path=Path("/lnm/share/donottouch.sh"),
    dask_jobscript_template=relative_path_from_queens(
        "templates/jobscripts/jobscript_bruteforce.sh"
    ),
)
CHARON_CONFIG = ClusterConfig(
    name="charon",
    cluster_address="charon.bauv.unibw-muenchen.de",
    workload_manager="slurm",
    jobscript_template=relative_path_from_queens("templates/jobscripts/jobscript_charon.sh"),
    cluster_internal_address="192.168.2.253",
    default_python_path="$HOME/miniconda3/envs/queens/bin/python",
    cluster_script_path=Path(),
    dask_jobscript_template=relative_path_from_queens("templates/jobscripts/jobscript_charon.sh"),
)

CLUSTER_CONFIGS = {
    THOUGHT_CLUSTER_TYPE: THOUGHT_CONFIG,
    BRUTEFORCE_CLUSTER_TYPE: BRUTEFORCE_CONFIG,
    CHARON_CLUSTER_TYPE: CHARON_CONFIG,
}


# CLUSTER TESTS ------------------------------------------------------------------------------------


@pytest.fixture(name="cluster_user", scope="session")
def cluster_user_fixture(pytestconfig):
    """Username of cluster account to use for tests."""
    return pytestconfig.getoption("remote_user")


@pytest.fixture(name="cluster", scope="session")
def cluster_fixture(request):
    """Iterate over clusters.

    The actual parameterization is done on a per test basis which also
    defines the parameterized markers of the tests.
    """
    return request.param


@pytest.fixture(name="cluster_settings", scope="session")
def cluster_settings_fixture(cluster, cluster_user):
    """Hold all settings of cluster."""
    settings = CLUSTER_CONFIGS.get(cluster).dict()
    _logger.debug("raw cluster config: %s", settings)
    settings["cluster"] = cluster
    settings["cluster_user"] = cluster_user
    settings["connect_to_resource"] = cluster_user + '@' + settings["cluster_address"]
    return settings


@pytest.fixture(name="connect_to_resource", scope="session")
def connect_to_resource_fixture(cluster_settings):
    """Use for ssh connect to the cluster."""
    return cluster_settings["connect_to_resource"]


@pytest.fixture(name="baci_cluster_paths", scope="session")
def baci_cluster_paths_fixture(connect_to_resource):
    """Paths to executables on the clusters.

    Checks also for existence of the executables.
    """
    _, _, remote_home, _ = run_subprocess_remote(
        "echo ~",
        remote_connect=connect_to_resource,
        additional_error_message=f"Unable to identify home on remote.\n"
        f"Tried to connect to {connect_to_resource}.",
    )

    base_directory = Path(remote_home.rstrip()) / "workspace" / "build"

    path_to_executable = base_directory / "baci-release"
    path_to_post_processor = base_directory / "post_processor"
    path_to_post_ensight = base_directory / "post_ensight"

    def exists_on_remote(file_path):
        """Check for existence of a file on remote machine."""
        command_string = f'find {file_path}'
        run_subprocess_remote(
            command=command_string,
            remote_connect=connect_to_resource,
            additional_error_message=f"Could not find executable on {connect_to_resource}.\n"
            f"Was looking here: {file_path}",
        )

    exists_on_remote(path_to_executable)
    exists_on_remote(path_to_post_processor)
    exists_on_remote(path_to_post_ensight)

    baci_cluster_paths = {
        'path_to_executable': path_to_executable,
        'path_to_post_ensight': path_to_post_ensight,
        'path_to_post_processor': path_to_post_processor,
    }
    return baci_cluster_paths


@pytest.fixture(name="baci_example_expected_mean")
def baci_example_expected_mean_fixture():
    """Expected result for the BACI example."""
    result = np.array(
        [
            [0.0041549, 0.00138497, -0.00961201],
            [0.00138497, 0.00323159, -0.00961201],
            [0.00230828, 0.00323159, -0.00961201],
            [0.0041549, 0.00230828, -0.00961201],
            [0.00138497, 0.0041549, -0.00961201],
            [0.0041549, 0.00323159, -0.00961201],
            [0.00230828, 0.0041549, -0.00961201],
            [0.0041549, 0.0041549, -0.00961201],
            [0.00138497, 0.00138497, -0.00961201],
            [0.00323159, 0.00138497, -0.00961201],
            [0.00138497, 0.00230828, -0.00961201],
            [0.00230828, 0.00138497, -0.00961201],
            [0.00323159, 0.00230828, -0.00961201],
            [0.00230828, 0.00230828, -0.00961201],
            [0.00323159, 0.00323159, -0.00961201],
            [0.00323159, 0.0041549, -0.00961201],
        ]
    )
    return result


@pytest.fixture(name="baci_example_expected_var")
def baci_example_expected_var_fixture():
    """Expected variance for the BACI example."""
    result = np.array(
        [
            [3.19513506e-07, 3.55014593e-08, 2.94994460e-07],
            [3.55014593e-08, 1.93285820e-07, 2.94994460e-07],
            [9.86153027e-08, 1.93285820e-07, 2.94994460e-07],
            [3.19513506e-07, 9.86153027e-08, 2.94994460e-07],
            [3.55014593e-08, 3.19513506e-07, 2.94994460e-07],
            [3.19513506e-07, 1.93285820e-07, 2.94994460e-07],
            [9.86153027e-08, 3.19513506e-07, 2.94994460e-07],
            [3.19513506e-07, 3.19513506e-07, 2.94994460e-07],
            [3.55014593e-08, 3.55014593e-08, 2.94994460e-07],
            [1.93285820e-07, 3.55014593e-08, 2.94994460e-07],
            [3.55014593e-08, 9.86153027e-08, 2.94994460e-07],
            [9.86153027e-08, 3.55014593e-08, 2.94994460e-07],
            [1.93285820e-07, 9.86153027e-08, 2.94994460e-07],
            [9.86153027e-08, 9.86153027e-08, 2.94994460e-07],
            [1.93285820e-07, 1.93285820e-07, 2.94994460e-07],
            [1.93285820e-07, 3.19513506e-07, 2.94994460e-07],
        ]
    )
    return result


@pytest.fixture(name="baci_example_expected_output")
def baci_example_expected_output_fixture():
    """Expected outputs for the BACI example."""
    result = np.array(
        [
            [
                [0.00375521, 0.00125174, -0.00922795],
                [0.00125174, 0.00292072, -0.00922795],
                [0.00208623, 0.00292072, -0.00922795],
                [0.00375521, 0.00208623, -0.00922795],
                [0.00125174, 0.00375521, -0.00922795],
                [0.00375521, 0.00292072, -0.00922795],
                [0.00208623, 0.00375521, -0.00922795],
                [0.00375521, 0.00375521, -0.00922795],
                [0.00125174, 0.00125174, -0.00922795],
                [0.00292072, 0.00125174, -0.00922795],
                [0.00125174, 0.00208623, -0.00922795],
                [0.00208623, 0.00125174, -0.00922795],
                [0.00292072, 0.00208623, -0.00922795],
                [0.00208623, 0.00208623, -0.00922795],
                [0.00292072, 0.00292072, -0.00922795],
                [0.00292072, 0.00375521, -0.00922795],
            ],
            [
                [0.00455460, 0.00151820, -0.00999606],
                [0.00151820, 0.00354247, -0.00999606],
                [0.00253033, 0.00354247, -0.00999606],
                [0.00455460, 0.00253033, -0.00999606],
                [0.00151820, 0.00455460, -0.00999606],
                [0.00455460, 0.00354247, -0.00999606],
                [0.00253033, 0.00455460, -0.00999606],
                [0.00455460, 0.00455460, -0.00999606],
                [0.00151820, 0.00151820, -0.00999606],
                [0.00354247, 0.00151820, -0.00999606],
                [0.00151820, 0.00253033, -0.00999606],
                [0.00253033, 0.00151820, -0.00999606],
                [0.00354247, 0.00253033, -0.00999606],
                [0.00253033, 0.00253033, -0.00999606],
                [0.00354247, 0.00354247, -0.00999606],
                [0.00354247, 0.00455460, -0.00999606],
            ],
        ]
    )
    return result
