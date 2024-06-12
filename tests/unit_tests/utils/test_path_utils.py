"""Test path utils."""

from pathlib import Path, PurePath

import pytest

from queens.utils.path_utils import (
    PATH_TO_QUEENS,
    PATH_TO_SOURCE,
    check_if_path_exists,
    create_folder_if_not_existent,
    relative_path_from_queens,
    relative_path_from_source,
)


@pytest.fixture(name="path_to_queens")
def fixture_path_to_queens():
    """Path to QUEENS."""
    return str(Path(__file__).parents[3])


@pytest.fixture(name="path_to_pqueens")
def fixture_path_to_pqueens():
    """Path to queens."""
    return str(Path(__file__).parents[3] / 'queens')


def extract_last_dirs(path, num_dirs):
    """Extract the n=num_dirs last directories from the given path.

    Intended to cut off the stem of the queens directory.

    Args:
        path (str, Path): path from which to extract the last directories
        num_dirs (int): number of directories to extract
    Returns:
        Path: path with the extracted directories
    """
    path = Path(path).resolve()
    path = path.relative_to(path.parents[num_dirs - 1])

    return path


def test_path_to_pqueens(path_to_pqueens):
    """Test path to queens."""
    num_dirs = 4
    assert extract_last_dirs(path_to_pqueens, num_dirs) == extract_last_dirs(
        PATH_TO_SOURCE, num_dirs
    )


def test_path_to_queens(path_to_queens):
    """Test path to queens."""
    num_dirs = 3
    assert extract_last_dirs(path_to_queens, num_dirs) == extract_last_dirs(
        PATH_TO_QUEENS, num_dirs
    )


def test_check_if_path_exists():
    """Test if path exists."""
    current_folder = Path(__file__).parent
    assert check_if_path_exists(current_folder)


def test_check_if_path_exists_not_existing():
    """Test if path does not exist."""
    tmp_path = Path(__file__).parent / "not_existing"
    with pytest.raises(FileNotFoundError):
        check_if_path_exists(tmp_path)


def test_create_folder_if_not_existent(tmp_path):
    """Test if folder is created."""
    new_path = PurePath(tmp_path, "new/path")
    new_path = create_folder_if_not_existent(new_path)
    assert check_if_path_exists(new_path)


def test_relative_path_from_source():
    """Test relative path from queens."""
    current_folder = Path(__file__).parent
    path_from_pqueens = relative_path_from_source("../tests/unit_tests/utils")
    num_dirs = 6
    assert extract_last_dirs(path_from_pqueens, num_dirs) == extract_last_dirs(
        current_folder, num_dirs
    )


def test_relative_path_from_queens():
    """Test relative path from queens."""
    current_folder = Path(__file__).parent
    path_from_queens = relative_path_from_queens("tests/unit_tests/utils")
    num_dirs = 6
    assert extract_last_dirs(path_from_queens, num_dirs) == extract_last_dirs(
        current_folder, num_dirs
    )
