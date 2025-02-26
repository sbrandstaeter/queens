#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Test path utils."""

from pathlib import Path, PurePath

import pytest

from queens.utils.path import (
    PATH_TO_QUEENS,
    PATH_TO_SOURCE,
    check_if_path_exists,
    create_folder_if_not_existent,
    is_empty,
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
    return str(Path(__file__).parents[3] / "queens")


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


def test_is_empty_with_empty_string():
    """Expected True for an empty string."""
    assert is_empty("")


def test_is_empty_with_non_empty_string():
    """Expected False for a non-empty string."""
    assert not is_empty("not_empty")


def test_is_empty_with_path_object():
    """Expected False for a Path object."""
    assert not is_empty(Path("/some/path"))


def test_is_empty_with_empty_list():
    """Expected True for an empty list."""
    assert is_empty([])


def test_is_empty_with_non_empty_list():
    """Expected False for a non-empty list."""
    assert not is_empty(["item"])


def test_is_empty_with_invalid_type():
    """Expected TypeError for invalid type."""
    with pytest.raises(TypeError, match="paths must be a string, a Path object, or a list."):
        is_empty(123)  # Integer input to trigger TypeError
