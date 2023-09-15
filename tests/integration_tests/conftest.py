"""Collect fixtures used by the integration tests."""
import ast
import getpass
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import yaml

from queens.utils.path_utils import relative_path_from_queens
from queens.utils.remote_operations import RemoteConnection

_logger = logging.getLogger(__name__)

THOUGHT_CLUSTER_TYPE = "thought"
BRUTEFORCE_CLUSTER_TYPE = "bruteforce"
CHARON_CLUSTER_TYPE = "charon"


@dataclass(frozen=True)
class ClusterConfig:
    """Configuration data of cluster.

    Attributes:
        name (str):                         name of cluster
        host (str):                         hostname or ip address to reach cluster from network
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
    host: str
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
    host="129.187.58.22",
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
    host="bruteforce.lnm.ed.tum.de",
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
    host="charon.bauv.unibw-muenchen.de",
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
@pytest.fixture(name="user", scope="session")
def fixture_user():
    """Name of user calling the test suite."""
    return getpass.getuser()


@pytest.fixture(name="remote_user", scope="session")
def fixture_remote_user(pytestconfig):
    """Username of cluster account to use for tests."""
    return pytestconfig.getoption("remote_user")


@pytest.fixture(name="gateway", scope="session")
def fixture_gateway(pytestconfig):
    """Host and user for gateway connection (proxyjump)."""
    gateway = pytestconfig.getoption("gateway")

    if isinstance(gateway, str):
        # Parse the string as an abstract syntax tree
        ast_tree = ast.literal_eval(gateway)

        # Check if the result is a dictionary
        if isinstance(ast_tree, dict):
            gateway_dict = ast_tree
            _logger.debug("Successfully converted string to dictionary: %s", gateway_dict)
        else:
            _logger.debug("The string '%s' does not represent a dictionary.", gateway)
        return gateway_dict
    return gateway

    # gateway_host = pytestconfig.getoption("gateway_host")
    # gateway_user = pytestconfig.getoption("gateway_user")
    # if (gateway_host is None and gateway_user is not None) or (
    #    gateway_host is not None and gateway_user is None
    # ):
    #    raise ValueError(
    #        f"'gateway_host={gateway_host}' and 'gateway_user={gateway_user}'. "
    #        "Either both are 'None' or none are 'None'."
    #    )
    # return {"host": gateway_host, "user": gateway_user}


@pytest.fixture(name="cluster", scope="session")
def fixture_cluster(request):
    """Iterate over clusters.

    The actual parameterization is done on a per test basis which also
    defines the parameterized markers of the tests.
    """
    return request.param


@pytest.fixture(name="cluster_settings", scope="session")
def fixture_cluster_settings(
    cluster, remote_user, gateway, remote_python, remote_queens_repository
):
    """Hold all settings of cluster."""
    settings = CLUSTER_CONFIGS.get(cluster).dict()
    _logger.debug("raw cluster config: %s", settings)
    settings["cluster"] = cluster
    settings["user"] = remote_user
    settings["connect_to_resource"] = remote_user + '@' + settings["host"]
    # gateway_host = gateway_settings["host"]
    # # the settings dictionary serves to write the value into the yaml input file thus
    # # convert the None value to a yaml equivalent "null" value
    # settings["gateway_host"] = gateway_host if gateway_host is not None else "null"

    # gateway_user = gateway_settings["user"]
    # # the settings dictionary serves to write the value into the yaml input file thus
    # # convert the None value to a yaml equivalent "null" value
    # settings["gateway_user"] = gateway_user if gateway_user is not None else "null"

    # gateway_connection = (
    #     None if gateway_host is None else Connection(host=gateway_host, user=gateway_user)
    # )

    # save the settings for the remote connection in string of yaml format to make it more flexible
    # for parsing it into the yaml input file
    settings["remote_connection"] = "  " + yaml.dump(
        {
            "host": settings["host"],
            "user": remote_user,
            "gateway": gateway,
            "remote_python": remote_python,
            "remote_queens_repository": remote_queens_repository,
        }
    ).replace("\n", "\n  ")

    #     Connection(
    #     host=settings["host"], user=remote_user, gateway=gateway_connection
    # )
    return settings


@pytest.fixture(name="connect_to_resource", scope="session")
def fixture_connect_to_resource(cluster_settings):
    """Use for ssh connect to the cluster."""
    return cluster_settings["connect_to_resource"]


@pytest.fixture(name="remote_python", scope="session")
def fixture_remote_python(pytestconfig):
    """Path to Python environment on remote host."""
    return pytestconfig.getoption("remote_python")


#    return pytestconfig.getoption("remote_python", default=cluster_settings["default_python_path"])


@pytest.fixture(name="remote_connection", scope="session")
def fixture_remote_connection(cluster_settings):
    """Fabric connection to remote."""
    # reconstruct the dict from the yaml
    remote_connection_config = yaml.safe_load(cluster_settings["remote_connection"])
    remote_connection = RemoteConnection(**remote_connection_config)
    return remote_connection


@pytest.fixture(name="remote_queens_repository", scope="session")
def fixture_remote_queens_repository(pytestconfig):
    """Path to queens repository on remote host."""
    remote_queens = pytestconfig.getoption("remote_queens_repository", skip=True)
    return remote_queens


@pytest.fixture(name="baci_cluster_paths", scope="session")
def fixture_baci_cluster_paths(remote_connection):
    """Paths to executables on the clusters.

    Checks also for existence of the executables.
    """
    result = remote_connection.run("echo ~", in_stream=False)
    remote_home = Path(result.stdout.rstrip())

    base_directory = remote_home / "workspace" / "build"

    path_to_executable = base_directory / "baci-release"
    path_to_post_processor = base_directory / "post_processor"
    path_to_post_ensight = base_directory / "post_ensight"

    def exists_on_remote(file_path):
        """Check for existence of a file on remote machine."""
        find_result = remote_connection.run(f'find {file_path}', in_stream=False)
        return Path(find_result.stdout.rstrip())

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
def fixture_baci_example_expected_mean():
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
def fixture_baci_example_expected_var():
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
def fixture_baci_example_expected_output():
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
