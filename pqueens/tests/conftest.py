"""Configuration module for the entire test suite (highest level)."""
import getpass
import logging
import socket
from pathlib import Path

import pytest

from pqueens.utils import config_directories
from pqueens.utils.logger_settings import reset_logging
from pqueens.utils.path_utils import relative_path_from_pqueens, relative_path_from_queens

_logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    """Add pytest options."""
    # default remote_user is same as local_user
    local_user = getpass.getuser()
    parser.addoption("--remote-user", action="store", default=local_user)
    parser.addoption("--remote-python", action="store", default=None)
    parser.addoption("--remote-queens-repository", action="store", default="null")


def pytest_collection_modifyitems(items):
    """Automatically add pytest markers based on testpath."""
    for item in items:
        if "benchmarks/" in item.nodeid:
            item.add_marker(pytest.mark.benchmark)
        elif "integration_tests/python/" in item.nodeid:
            item.add_marker(pytest.mark.integration_tests)
        elif "integration_tests/baci/" in item.nodeid:
            item.add_marker(pytest.mark.integration_tests_baci)
        elif "unit_tests/" in item.nodeid:
            item.add_marker(pytest.mark.unit_tests)


NAME_OF_HOST = socket.gethostname()


@pytest.fixture(name="hostname", scope="session")
def hostname_fixture(name_of_host=NAME_OF_HOST):
    """Hostname calling the test suite."""
    _logger.debug("Tests are run on: %s", name_of_host)
    return name_of_host


@pytest.fixture(name="global_mock_local_base_dir", autouse=True)
def global_mock_local_base_dir_fixture(monkeypatch, tmp_path):
    """Mock the local base directory for all tests.

    This is necessary to keep the base directory of a user clean from
    testing data. pytest temp_path supplies a perfect location for this
    (see pytest docs).
    """

    def mock_local_base_dir():
        return tmp_path

    monkeypatch.setattr(config_directories, "local_base_directory", mock_local_base_dir)
    _logger.debug("Mocking of local base dir was successful.")
    _logger.debug("local base dir is mocked to: %s", config_directories.local_base_directory())


@pytest.fixture(name="mock_value_experiments_base_folder_name", scope="session")
def mock_value_experiments_base_folder_name_fixture():
    """Value to mock the experiments base folder name."""
    return "pytest"


@pytest.fixture(name="global_mock_experiments_base_folder_name", autouse=True)
def global_mock_experiments_base_folder_name_fixture(
    mock_value_experiments_base_folder_name, monkeypatch
):
    """Mock the name of the folders containing experiments in base directory.

    Note that locally, this adds on top of the
    global_mock_local_base_dir
    """
    monkeypatch.setattr(
        config_directories, "EXPERIMENTS_BASE_FOLDER_NAME", mock_value_experiments_base_folder_name
    )
    _logger.debug("Mocking of EXPERIMENTS_BASE_FOLDER_NAME was successful.")
    _logger.debug(
        "EXPERIMENTS_BASE_FOLDER_NAME is mocked to: %s",
        config_directories.EXPERIMENTS_BASE_FOLDER_NAME,
    )


@pytest.fixture(name="inputdir", scope='session')
def inputdir_fixture():
    """Return the path to the json input-files of the function test."""
    input_files_path = relative_path_from_pqueens("tests/integration_tests/queens_input_files")
    return input_files_path


@pytest.fixture(name="third_party_inputs", scope='session')
def third_party_inputs_fixture():
    """Return the path to the json input-files of the function test."""
    input_files_path = relative_path_from_pqueens("tests/integration_tests/third_party_input_files")
    return input_files_path


@pytest.fixture(name="config_dir", scope='session')
def config_dir_fixture():
    """Return the path to the json input-files of the function test."""
    config_dir_path = relative_path_from_queens("config")
    return config_dir_path


@pytest.fixture(name="baci_link_paths", scope="session")
def baci_link_paths_fixture(config_dir):
    """Set symbolic links for baci on testing machine."""
    baci = config_dir / 'baci-release'
    post_ensight = config_dir / 'post_ensight'
    post_processor = config_dir / 'post_processor'
    return baci, post_ensight, post_processor


@pytest.fixture(name="baci_source_paths_for_gitlab_runner", scope="session")
def baci_source_paths_for_gitlab_runner_fixture():
    """Set symbolic links for baci on testing machine."""
    home = Path.home()
    src_baci = home / 'workspace/build/baci-release'
    src_post_ensight = home / 'workspace/build/post_ensight'
    src_post_processor = home / 'workspace/build/post_processor'
    return src_baci, src_post_ensight, src_post_processor


@pytest.fixture(name="example_simulator_fun_dir", scope='session')
def example_simulator_fun_dir_fixture():
    """Return the path to the example simulator functions."""
    input_files_path = relative_path_from_pqueens(
        "tests/integration_tests/example_simulator_functions"
    )
    return input_files_path


def pytest_sessionfinish():
    """Register a hook to suppress logging errors after the session."""
    logging.raiseExceptions = False


@pytest.fixture(name="reset_loggers", autouse=True)
def fixture_reset_logger():
    """Reset loggers.

    This fixture is called at every test due to `autouse=True`. It acts
    as a generator and allows us to close all loggers after each test.
    This should avoid duplicate logger output.
    """
    # Do the test.
    yield

    # Test is done, now reset the loggers.
    reset_logging()
