"""Collect fixtures used by the integration tests."""

import getpass
import logging
import shutil
from pathlib import Path

import numpy as np
import pytest

from pqueens.schedulers.cluster_scheduler import (
    BRUTEFORCE_CLUSTER_TYPE,
    CHARON_CLUSTER_TYPE,
    DEEP_CLUSTER_TYPE,
)
from pqueens.utils import config_directories
from pqueens.utils.io_utils import load_result
from pqueens.utils.manage_singularity import SingularityManager
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
def cluster_address(cluster):
    """Use for ssh connect to the cluster."""
    if cluster in (DEEP_CLUSTER_TYPE, BRUTEFORCE_CLUSTER_TYPE):
        address = cluster + '.lnm.ed.tum.de'
    elif cluster == CHARON_CLUSTER_TYPE:
        address = cluster + '.bauv.unibw-muenchen.de'
    return address


@pytest.fixture(scope="session")
def connect_to_resource(cluster_user, cluster_address):
    """Use for ssh connect to the cluster."""
    return cluster_user + '@' + cluster_address


@pytest.fixture(scope="session")
def cluster_singularity_ip(cluster):
    """Identify IP address of cluster."""
    if cluster == DEEP_CLUSTER_TYPE:
        cluster_singularity_ip = '129.187.58.20'
    elif cluster == BRUTEFORCE_CLUSTER_TYPE:
        cluster_singularity_ip = '10.10.0.1'
    elif cluster == CHARON_CLUSTER_TYPE:
        cluster_singularity_ip = '192.168.1.253'
    else:
        cluster_singularity_ip = None
    return cluster_singularity_ip


@pytest.fixture(scope="session")
def cluster_queens_base_dir(connect_to_resource):
    """Hold base directory for queens on cluster."""
    return config_directories.remote_base_directory(remote_connect=connect_to_resource)


@pytest.fixture(scope="session")
def cluster_queens_testing_folder(mock_value_experiments_base_folder_name, cluster_queens_base_dir):
    """Hold base directory for experiment data of tests."""
    return cluster_queens_base_dir / mock_value_experiments_base_folder_name


@pytest.fixture(scope="session")
def cluster_native_queens_testing_folder(mock_value_experiments_base_folder_name):
    """Hold base directory for experiment data of tests."""
    return config_directories.local_base_directory() / mock_value_experiments_base_folder_name


@pytest.fixture(scope="session")
def cluster_path_to_singularity(cluster_queens_base_dir):
    """Folder on cluster where to put the singularity file."""
    return cluster_queens_base_dir


@pytest.fixture(scope="session")
def prepare_cluster_testing_environment(connect_to_resource, cluster_queens_testing_folder):
    """Create a clean testing environment on the cluster."""
    # remove old folder
    _logger.info(
        "Delete testing folder %s on %s", cluster_queens_testing_folder, connect_to_resource
    )
    command_string = f'rm -rfv {cluster_queens_testing_folder}'
    _, _, stdout, _ = run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_connect=connect_to_resource,
    )
    _logger.info(stdout)

    # create generic testing folder
    _logger.info(
        "Create testing folder %s on %s", cluster_queens_testing_folder, connect_to_resource
    )
    command_string = f'mkdir -v -p {cluster_queens_testing_folder}'
    _, _, stdout, _ = run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_connect=connect_to_resource,
    )
    _logger.info(stdout)

    return True


@pytest.fixture(scope="session")
def prepare_singularity(
    connect_to_resource,
    cluster_queens_base_dir,
    prepare_cluster_testing_environment,
):
    """Build singularity based on the code during test invocation.

    **WARNING:** Needs to be done AFTER *prepare_cluster_testing_environment*
    to make sure cluster testing folder is clean and existing.
    """
    if not prepare_cluster_testing_environment:
        raise RuntimeError("Testing environment on cluster not successful.")

    singularity_manager = SingularityManager(
        singularity_path=cluster_queens_base_dir,
        singularity_bind=None,
        input_file=None,
        remote=True,
        remote_connect=connect_to_resource,
    )
    # singularity_manager.check_singularity_system_vars()
    singularity_manager.prepare_singularity_files()
    return True


@pytest.fixture(scope="session")
def cluster_testsuite_settings(
    cluster,
    cluster_user,
    cluster_address,
    connect_to_resource,
    prepare_singularity,
    cluster_singularity_ip,
):
    """Collect settings needed for cluster tests with singularity."""
    if not prepare_singularity:
        raise RuntimeError(
            "Preparation of singularity for cluster failed."
            "Make sure to prepare singularity image before using this fixture. "
        )
    cluster_testsuite_settings = {}
    cluster_testsuite_settings["cluster"] = cluster
    cluster_testsuite_settings["cluster_user"] = cluster_user
    cluster_testsuite_settings["cluster_address"] = cluster_address
    cluster_testsuite_settings["connect_to_resource"] = connect_to_resource
    cluster_testsuite_settings["singularity_remote_ip"] = cluster_singularity_ip

    return cluster_testsuite_settings


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
def prepare_cluster_testing_environment_native(cluster_native_queens_testing_folder):
    """Create a clean testing environment."""
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
    cluster_user, prepare_cluster_testing_environment_native
):  # pylint: disable=unused-argument
    """Paths to baci for native cluster tests."""
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
