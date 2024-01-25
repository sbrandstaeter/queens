"""Test copying with rsync."""
import filecmp

import pytest

from queens.utils.rsync import rsync


@pytest.fixture(name="files_to_copy")
def fixture_files_to_copy():
    """Files to copy."""
    return ["fileA", "fileB"]


@pytest.fixture(name="_create_source_files")
def fixture_create_source_files(source_path, files_to_copy):
    """Create source files."""
    for file in files_to_copy:
        (source_path / file).write_text(file)


@pytest.fixture(name="source_path")
def fixture_source_path(tmp_path):
    """Source path."""
    source_path = tmp_path / "source"
    source_path.mkdir()
    return source_path


@pytest.fixture(name="destination_path")
def fixture_destination_path(tmp_path):
    """Destination path."""
    destination_path = tmp_path / "destination"
    return destination_path


def test_single_file(_create_source_files, source_path, destination_path, files_to_copy):
    """Test copy a single file."""
    source_file = source_path / files_to_copy[0]
    rsync(source_file, destination_path)
    assert filecmp.cmp(destination_path / files_to_copy[0], source_file)


def test_multiple_files(_create_source_files, source_path, destination_path, files_to_copy):
    """Test copy a list of files."""
    source_files = [source_path / file for file in files_to_copy]
    rsync(source_files, destination_path)
    match, mismatch, errors = filecmp.cmpfiles(destination_path, source_path, common=files_to_copy)
    assert len(match) == len(files_to_copy)  # all files are copied
    assert not mismatch  # no mismatches
    assert not errors  # no errors


def test_directory(_create_source_files, source_path, destination_path, files_to_copy):
    """Test copy a directory."""
    rsync(source_path, destination_path)
    match, mismatch, errors = filecmp.cmpfiles(
        destination_path / source_path.name, source_path, common=files_to_copy
    )
    assert len(match) == len(files_to_copy)  # all files are copied
    assert not mismatch  # no mismatches
    assert not errors  # no errors
