"""Configuration module for the entire test suite (highest level)."""
import pathlib

import pytest

from pqueens.utils import config_directories, manage_singularity
from pqueens.utils.path_utils import relative_path_from_pqueens, relative_path_from_queens


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


@pytest.fixture(autouse=True)
def global_mock_abs_singularity_image_path(monkeypatch):
    """Mock the absolute singularity image path.

    The singularity image path depends on local_base_dir() which is
    mocked globally and per test. This would however mean that every
    test that needs singularity has to built the image again. Because
    one test does not know about the other ones. To prevent this we
    store the standard image path that is used by the user here and make
    sure that this path is used in all tests by mocking the respective
    variable globally. This way the image has to be built only once.
    """
    mock_abs_singularity_image_path = manage_singularity.ABS_SINGULARITY_IMAGE_PATH
    monkeypatch.setattr(
        manage_singularity, "ABS_SINGULARITY_IMAGE_PATH", mock_abs_singularity_image_path
    )


@pytest.fixture(autouse=True)
def global_mock_local_base_dir(monkeypatch, tmp_path):
    """Mock the local base directory for all tests.

    This is necessary to keep the base directory of a user clean from
    testing data. pytest temp_path supplies a perfect location for this
    (see pytest docs).
    """

    def mock_local_base_dir():
        return tmp_path

    monkeypatch.setattr(config_directories, "local_base_dir", mock_local_base_dir)


@pytest.fixture(scope='session')
def inputdir():
    """Return the path to the json input-files of the function test."""
    input_files_path = relative_path_from_pqueens("tests/integration_tests/queens_input_files")
    return input_files_path


@pytest.fixture(scope='session')
def third_party_inputs():
    """Return the path to the json input-files of the function test."""
    input_files_path = relative_path_from_pqueens("tests/integration_tests/third_party_input_files")
    return input_files_path


@pytest.fixture(scope='session')
def config_dir():
    """Return the path to the json input-files of the function test."""
    config_dir_path = relative_path_from_queens("config")
    return config_dir_path


@pytest.fixture(scope="session")
def baci_link_paths(config_dir):
    """Set symbolic links for baci on testing machine."""
    baci = str(pathlib.Path(config_dir).joinpath('baci-release'))
    post_drt_monitor = str(pathlib.Path(config_dir).joinpath('post_drt_monitor'))
    post_drt_ensight = str(pathlib.Path(config_dir).joinpath('post_drt_ensight'))
    post_processor = str(pathlib.Path(config_dir).joinpath('post_processor'))
    return baci, post_drt_monitor, post_drt_ensight, post_processor


@pytest.fixture(scope="session")
def baci_source_paths_for_gitlab_runner():
    """Set symbolic links for baci on testing machine."""
    home = pathlib.Path.home()
    src_baci = pathlib.Path.joinpath(home, 'workspace/build/baci-release')
    src_drt_monitor = pathlib.Path.joinpath(home, 'workspace/build/post_drt_monitor')
    src_post_drt_ensight = pathlib.Path.joinpath(home, 'workspace/build/post_drt_ensight')
    src_post_processor = pathlib.Path.joinpath(home, 'workspace/build/post_processor')
    return src_baci, src_drt_monitor, src_post_drt_ensight, src_post_processor


@pytest.fixture(scope='session')
def example_simulator_fun_dir():
    """Return the path to the example simulator functions."""
    input_files_path = relative_path_from_pqueens(
        "tests/integration_tests/example_simulator_functions"
    )
    return input_files_path
