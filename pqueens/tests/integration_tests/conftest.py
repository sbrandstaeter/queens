"""Collect fixtures used by the integration tests."""

import getpass
import pathlib

import pytest

from pqueens.schedulers.cluster_scheduler import (
    BRUTEFORCE_SCHEDULER_TYPE,
    CHARON_SCHEDULER_TYPE,
    DEEP_SCHEDULER_TYPE,
)
from pqueens.utils.manage_singularity import SingularityManager
from pqueens.utils.run_subprocess import run_subprocess


# CLUSTER TESTS ------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def user():
    """Name of user calling the test suite."""
    return getpass.getuser()


@pytest.fixture(scope="session")
def cluster_user(user):
    """Username of cluster account to use for tests."""
    # user who called the test suite
    # gitlab-runner has to run simulation as different user on cluster everyone else should use
    # account with same name
    if user == "gitlab-runner":
        cluster_user = "queens"
    else:
        cluster_user = user
    return cluster_user


@pytest.fixture(scope="session")
def cluster(request):
    """Helper fixture to iterate over clusters.

    The actual parameterization is done on a per test basis which also
    defines the parameterized markers of the tests.
    """
    return request.param


@pytest.fixture(scope="session")
def cluster_address(cluster):
    """String used for ssh connect to the cluster."""
    if cluster == "deep" or cluster == "bruteforce":
        address = cluster + '.lnm.ed.tum.de'
    elif cluster == "charon":
        address = cluster + '.bauv.unibw-muenchen.de'
    return address


@pytest.fixture(scope="session")
def connect_to_resource(cluster_user, cluster_address):
    """String used for ssh connect to the cluster."""
    connect_to_resource = cluster_user + '@' + cluster_address
    return connect_to_resource


@pytest.fixture(scope="session")
def cluster_singularity_ip(cluster):
    """Identify IP address of cluster."""
    if cluster == "deep":
        cluster_singularity_ip = '129.187.58.20'
    elif cluster == "bruteforce":
        cluster_singularity_ip = '10.10.0.1'
    elif cluster == "charon":
        cluster_singularity_ip = '192.168.1.253'
    else:
        cluster_singularity_ip = None
    return cluster_singularity_ip


@pytest.fixture(scope="session")
def scheduler_type(cluster):
    """Switch type of scheduler according to cluster."""
    if cluster == "deep":
        scheduler_type = DEEP_SCHEDULER_TYPE
    elif cluster == "bruteforce":
        scheduler_type = BRUTEFORCE_SCHEDULER_TYPE
    elif cluster == "charon":
        scheduler_type = CHARON_SCHEDULER_TYPE
    return scheduler_type


@pytest.fixture(scope="session")
def cluster_queens_testing_folder(cluster_user):
    """Generic folder on cluster for testing."""
    cluster_queens_testing_folder = pathlib.Path("/home", cluster_user, "queens-testing")
    return cluster_queens_testing_folder


@pytest.fixture(scope="session")
def cluster_path_to_singularity(cluster_queens_testing_folder):
    """Folder on cluster where to put the singularity file."""
    cluster_path_to_singularity = cluster_queens_testing_folder.joinpath("singularity")
    return cluster_path_to_singularity


@pytest.fixture(scope="session")
def prepare_cluster_testing_environment(
    cluster_user, cluster_address, cluster_queens_testing_folder, cluster_path_to_singularity
):
    """Create a clean testing environment on the cluster."""
    # remove old folder
    print(f"Delete testing folder from {cluster_address}")
    command_string = f'rm -rfv {cluster_queens_testing_folder}'
    returncode, pid, stdout, stderr = run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_user=cluster_user,
        remote_address=cluster_address,
    )
    print(stdout)
    if returncode:
        raise Exception(stderr)

    # create generic testing folder
    print(f"Create testing folder on {cluster_address}")
    command_string = f'mkdir -v -p {cluster_queens_testing_folder}'
    _, _, stdout, _ = run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_user=cluster_user,
        remote_address=cluster_address,
    )
    print(stdout)

    # create folder for singularity
    print(f"Create folder for singularity image on {cluster_address}")
    command_string = f'mkdir -v -p {cluster_path_to_singularity}'
    _, _, stdout, _ = run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_user=cluster_user,
        remote_address=cluster_address,
    )
    print(stdout)

    return True


@pytest.fixture(scope="session")
def prepare_singularity(
    connect_to_resource,
    cluster_path_to_singularity,
    prepare_cluster_testing_environment,
):
    """Build singularity based on the code during test invocation.

    WARNING: needs to be done AFTER prepare_cluster_testing_environment to make sure cluster testing
     folder is clean and existing
    """
    if not prepare_cluster_testing_environment:
        raise RuntimeError("Testing environment on cluster not successful.")

    remote_flag = True
    singularity_manager = SingularityManager(
        singularity_path=cluster_path_to_singularity,
        singularity_bind=None,
        input_file=None,
        remote=remote_flag,
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
    cluster_queens_testing_folder,
    cluster_path_to_singularity,
    prepare_singularity,
    scheduler_type,
    cluster_singularity_ip,
):
    """Collection of settings needed for all cluster tests."""
    if not prepare_singularity:
        raise RuntimeError(
            "Preparation of singularity for cluster failed."
            "Make sure to prepare singularity image before using this fixture. "
        )
    cluster_testsuite_settings = dict()
    cluster_testsuite_settings["cluster"] = cluster
    cluster_testsuite_settings["cluster_user"] = cluster_user
    cluster_testsuite_settings["cluster_address"] = cluster_address
    cluster_testsuite_settings["connect_to_resource"] = connect_to_resource
    cluster_testsuite_settings["cluster_queens_testing_folder"] = cluster_queens_testing_folder
    cluster_testsuite_settings["cluster_path_to_singularity"] = cluster_path_to_singularity
    cluster_testsuite_settings["scheduler_type"] = scheduler_type
    cluster_testsuite_settings["singularity_remote_ip"] = cluster_singularity_ip

    return cluster_testsuite_settings


@pytest.fixture(scope="session")
def baci_cluster_paths(cluster_user, cluster_address):
    """Paths to executables on the clusters.

    Checks also for existance of the executables.
    """
    path_to_executable = pathlib.Path("/home", cluster_user, "workspace", "build", "baci-release")

    path_to_drt_monitor = pathlib.Path(
        "/home", cluster_user, "workspace", "build", "post_drt_monitor"
    )

    path_to_post_processor = pathlib.Path(
        "/home", cluster_user, "workspace", "build", "post_processor"
    )

    path_to_drt_ensight = pathlib.Path(
        "/home", cluster_user, "workspace", "build", "post_drt_ensight"
    )

    command_string = f'find {path_to_executable}'
    run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_user=cluster_user,
        remote_address=cluster_address,
        additional_error_message=f"Could not find executable on {cluster_address}.\n"
        f"Was looking here: {path_to_executable}",
    )

    command_string = f'find {path_to_drt_monitor}'
    run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_user=cluster_user,
        remote_address=cluster_address,
        additional_error_message=f"Could not find postprocessor on {cluster_address}.\n"
        f"Was looking here: {path_to_drt_monitor}",
    )

    command_string = f'find {path_to_drt_ensight}'
    run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_user=cluster_user,
        remote_address=cluster_address,
        additional_error_message=f"Could not find postprocessor on {cluster_address}.\n"
        f"Was looking here: {path_to_drt_ensight}",
    )

    command_string = f'find {path_to_post_processor}'
    run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_user=cluster_user,
        remote_address=cluster_address,
        additional_error_message=f"Could not find postprocessor on {cluster_address}.\n"
        f"Was looking here: {path_to_post_processor}",
    )

    baci_cluster_paths = dict(
        path_to_executable=path_to_executable,
        path_to_drt_monitor=path_to_drt_monitor,
        path_to_drt_ensight=path_to_drt_ensight,
        path_to_post_processor=path_to_post_processor,
    )
    return baci_cluster_paths
