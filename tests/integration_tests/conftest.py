"""Collect fixtures used by the integration tests."""

import getpass
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import yaml

from queens.example_simulator_functions.park91a import X3, X4, park91a_hifi_on_grid
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
    cluster_internal_address: str | None
    default_python_path: str
    cluster_script_path: Path
    dask_jobscript_template: Path
    queue: Optional[str | None] = None

    dict = asdict


THOUGHT_CONFIG = ClusterConfig(
    name="thought",
    host="129.187.58.22",
    workload_manager="slurm",
    queue="normal",
    jobscript_template=relative_path_from_queens("templates/jobscripts/jobscript_thought.sh"),
    cluster_internal_address=None,
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
    """String of a dictionary that defines gateway connection (proxyjump)."""
    gateway = pytestconfig.getoption("gateway")
    if isinstance(gateway, str):
        gateway = json.loads(gateway)
    return gateway


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
    settings["remote_python"] = remote_python
    settings["remote_queens_repository"] = remote_queens_repository

    if gateway is None:
        # None is equal to null in yaml
        settings["gateway"] = "null"
    elif isinstance(gateway, dict):
        # the gateway settings should be supplied via a dict:
        # save the settings in string of yaml format to make it more flexible for parsing it into
        # the yaml input file (we don't know which keywords the user used to supply the settings)

        # in the yaml file this dict is already two level indented: add four spaces before each line
        indentation = 4 * " "
        settings["gateway"] = (
            "\n" + indentation + yaml.dump(gateway).replace("\n", "\n" + indentation)
        )
    else:
        raise ValueError(f"Cannot handle gateway information {gateway} of type {type(gateway)}.")
    return settings


@pytest.fixture(name="remote_python", scope="session")
def fixture_remote_python(pytestconfig):
    """Path to Python environment on remote host."""
    return pytestconfig.getoption("remote_python")


@pytest.fixture(name="remote_connection", scope="session")
def fixture_remote_connection(cluster_settings, gateway):
    """Fabric connection to remote."""
    return RemoteConnection(
        host=cluster_settings["host"],
        user=cluster_settings["user"],
        remote_python=cluster_settings["remote_python"],
        remote_queens_repository=cluster_settings["remote_queens_repository"],
        gateway=gateway,
    )


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


@pytest.fixture(name="_create_experimental_data_park91a_hifi_on_grid")
def fixture_create_experimental_data_park91a_hifi_on_grid(tmp_path):
    """Create experimental data."""
    # Fix random seed
    np.random.seed(seed=1)

    # True input values
    x1 = 0.5
    x2 = 0.2

    y_vec = park91a_hifi_on_grid(x1, x2)

    # Artificial noise
    sigma_n = 0.001
    noise_vec = np.random.normal(loc=0, scale=sigma_n, size=(y_vec.size,))

    # Inverse crime: Add artificial noise to model output for the true value
    y_fake = y_vec + noise_vec

    # write fake data to csv
    data_dict = {
        'x3': X3,
        'x4': X4,
        'y_obs': y_fake,
    }
    experimental_data_path = tmp_path / 'experimental_data.csv'
    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.to_csv(experimental_data_path, index=False)


# fixtures for Elementary Effects Sobol tests
@pytest.fixture(name="expected_result_mu")
def fixture_expected_result_mu():
    """Mu result fixture."""
    expected_result_mu = np.array(
        [
            25.8299150077341,
            19.28297176050532,
            -14.092164789704626,
            5.333475971922498,
            -11.385141403296364,
            13.970208961715421,
            -3.0950202483238303,
            0.6672725255532903,
            7.2385092339309445,
            -7.7664016980947075,
        ]
    )
    return expected_result_mu


@pytest.fixture(name="expected_result_mu_star")
def fixture_expected_result_mu_star():
    """Mu star result fixture."""
    expected_result_mu_star = np.array(
        [
            29.84594504725642,
            21.098173537614855,
            16.4727722348437,
            26.266876218598668,
            16.216603266281044,
            18.051629859410895,
            3.488313966697564,
            2.7128638920479147,
            7.671230484535577,
            10.299932289624746,
        ]
    )
    return expected_result_mu_star


@pytest.fixture(name="expected_result_sigma")
def fixture_expected_result_sigma():
    """Sigma result fixture."""
    expected_result_sigma = np.array(
        [
            53.88783786787971,
            41.02192670857979,
            29.841807478998156,
            43.33349033575829,
            29.407676882180404,
            31.679653142831512,
            5.241491105224932,
            4.252334015139214,
            10.38274186974731,
            18.83046700807382,
        ]
    )
    return expected_result_sigma
