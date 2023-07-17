"""Collect fixtures used by the integration tests."""

import getpass
import logging
import shutil
from pathlib import Path

import numpy as np
import pytest

from pqueens.schedulers.cluster_scheduler import CLUSTER_CONFIGS
from pqueens.utils import config_directories
from pqueens.utils.io_utils import load_result
from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


# CLUSTER TESTS ------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def user():
    """Name of user calling the test suite."""
    return getpass.getuser()


@pytest.fixture(scope="session")
def cluster_user(user, hostname):
    """Username of cluster account to use for tests."""
    # user who called the test suite
    # gitlab-runner has to run simulation as different user on cluster everyone else should use
    # account with same name
    if user == "gitlab-runner" and (hostname not in ["master.service", "login.cluster"]):
        cluster_user = "queens"
    else:
        cluster_user = user
    return cluster_user


@pytest.fixture(scope="session")
def cluster(request):
    """Iterate over clusters.

    The actual parameterization is done on a per test basis which also
    defines the parameterized markers of the tests.
    """
    return request.param


@pytest.fixture(scope="session")
def cluster_settings(cluster, cluster_user):
    """Hold all settings of cluster."""
    settings = CLUSTER_CONFIGS.get(cluster).dict()
    _logger.debug("raw cluster config: %s", settings)
    settings["cluster"] = cluster
    settings["cluster_user"] = cluster_user
    settings["connect_to_resource"] = cluster_user + '@' + settings["cluster_address"]
    return settings


@pytest.fixture(scope="session")
def connect_to_resource(cluster_settings):
    """Use for ssh connect to the cluster."""
    return cluster_settings["connect_to_resource"]


@pytest.fixture(scope="session")
def baci_cluster_paths(connect_to_resource):
    """Paths to executables on the clusters.

    Checks also for existence of the executables.
    """
    base_directory = config_directories.remote_home(connect_to_resource) / "workspace" / "build"

    path_to_executable = base_directory / "baci-release"
    path_to_post_processor = base_directory / "post_processor"
    path_to_drt_ensight = base_directory / "post_ensight"

    def exists_on_remote(file_path):
        """Check for existence of a file on remote machine."""
        command_string = f'find {file_path}'
        run_subprocess(
            command_string=command_string,
            subprocess_type='remote',
            remote_connect=connect_to_resource,
            additional_error_message=f"Could not find executable on {connect_to_resource}.\n"
            f"Was looking here: {file_path}",
        )

    exists_on_remote(path_to_executable)
    exists_on_remote(path_to_post_processor)
    exists_on_remote(path_to_drt_ensight)

    baci_cluster_paths = {
        'path_to_executable': path_to_executable,
        'path_to_drt_ensight': path_to_drt_ensight,
        'path_to_post_processor': path_to_post_processor,
    }
    return baci_cluster_paths


@pytest.fixture(scope="session")
def prepare_cluster_testing_environment_native(mock_value_experiments_base_folder_name):
    """Create a clean testing environment."""
    cluster_native_queens_testing_folder = (
        config_directories.local_base_directory() / mock_value_experiments_base_folder_name
    )
    if (
        cluster_native_queens_testing_folder.exists()
        and cluster_native_queens_testing_folder.is_dir()
    ):
        _logger.info("Delete testing folder")
        shutil.rmtree(cluster_native_queens_testing_folder)

    _logger.info("Create testing folder")
    cluster_native_queens_testing_folder.mkdir(parents=True, exist_ok=True)

    return True


# prepare_cluster_testing_environment_native is passed on purpose to force its creation
@pytest.fixture(scope="session")
def baci_cluster_paths_native(
    cluster_user, cluster_settings, prepare_cluster_testing_environment_native
):  # pylint: disable=unused-argument
    """Paths to baci for native cluster tests."""
    cluster_address = cluster_settings["cluster_address"]
    path_to_executable = Path(
        "/home", cluster_user, "workspace_for_queens", "build", "baci-release"
    )
    if not path_to_executable.is_file():
        raise RuntimeError(
            f"Could not find executable on {cluster_address}.\n"
            f"Was looking here: {path_to_executable}"
        )

    path_to_post_ensight = Path(
        "/home", cluster_user, "workspace_for_queens", "build", "post_ensight"
    )
    if not path_to_post_ensight.is_file():
        raise RuntimeError(
            f"Could not find postprocessor on {cluster_address}.\n"
            f"Was looking here: {path_to_post_ensight}"
        )

    baci_cluster_paths_native = {
        'path_to_executable': path_to_executable,
        'path_to_post_ensight': path_to_post_ensight,
    }
    return baci_cluster_paths_native


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
def name_baci_example_expected_var():
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
