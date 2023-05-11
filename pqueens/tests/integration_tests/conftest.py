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
    path_to_drt_monitor = base_directory / "post_drt_monitor"
    path_to_post_processor = base_directory / "post_processor"
    path_to_drt_ensight = base_directory / "post_drt_ensight"

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
    exists_on_remote(path_to_drt_monitor)
    exists_on_remote(path_to_post_processor)
    exists_on_remote(path_to_drt_ensight)

    baci_cluster_paths = {
        'path_to_executable': path_to_executable,
        'path_to_drt_monitor': path_to_drt_monitor,
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

    path_to_drt_monitor = Path(
        "/home", cluster_user, "workspace_for_queens", "build", "post_drt_monitor"
    )
    if not path_to_drt_monitor.is_file():
        raise RuntimeError(
            f"Could not find postprocessor on {cluster_address}.\n"
            f"Was looking here: {path_to_drt_monitor}"
        )

    baci_cluster_paths_native = {
        'path_to_executable': path_to_executable,
        'path_to_drt_monitor': path_to_drt_monitor,
    }
    return baci_cluster_paths_native


@pytest.fixture(scope="session")
def baci_elementary_effects_check_results():
    """Check results for baci elementary effects tests."""

    def check_results(result_file):
        """Check results for baci elementary effects tests.

        Args:
            result_file (path): path to result file
        """
        results = load_result(result_file)

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
            results["sensitivity_indices"]["mu_star_conf"],
            np.array([0.136631, 0.140794]),
            rtol=1.0e-3,
        )

    return check_results
