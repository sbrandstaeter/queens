"""Tests for numpy data processor."""

from pathlib import Path

import numpy as np
import pytest

from pqueens.data_processor.data_processor_numpy import DataProcessorNumpy


# ------ fixtures ----------
@pytest.fixture(name="data_path")
def fixture_data_path(tmp_path):
    """Create a dummy data path."""
    my_data_path = tmp_path / "my_data.npy"
    return my_data_path


@pytest.fixture(name="wrong_data_path")
def fixture_wrong_data_path(tmp_path):
    """Create a dummy data path."""
    my_data_path = tmp_path / "my_data.csv"
    return my_data_path


@pytest.fixture(name="dummy_data")
def fixture_dummy_data():
    """Create some dummy data."""
    data = np.array([[1, 2], [3, 4]])
    return data


@pytest.fixture(name="write_dummy_data")
def fixture_write_dummy_data(data_path, dummy_data):
    """Write the dummy data."""
    with open(data_path, "wb") as f:
        np.save(f, dummy_data)


@pytest.fixture(name="write_wrong_dummy_data")
def fixture_write_wrong_dummy_data(wrong_data_path, dummy_data):
    """Write the wrong dummy data."""
    with open(wrong_data_path, "w") as f:
        np.savetxt(f, dummy_data)


@pytest.fixture(name="default_data_processor_npy")
def fixture_default_data_processor_npy(data_path):
    """Dummy data processor npy."""
    file_name_identifier = "dummy"
    file_options_dict = {}
    files_to_be_deleted_regex_lst = []

    data_processor = DataProcessorNumpy(
        file_name_identifier,
        file_options_dict,
        files_to_be_deleted_regex_lst,
    )

    return data_processor


# ------ actual tests ------
def test_init():
    """Test initialization of npy data processor."""
    file_name_identifier = 'dummy'
    file_options_dict = {"dummy": "dummy"}
    files_to_be_deleted_regex_lst = ['abc']

    data_processor = DataProcessorNumpy(
        file_name_identifier,
        file_options_dict,
        files_to_be_deleted_regex_lst,
    )

    assert data_processor.files_to_be_deleted_regex_lst == files_to_be_deleted_regex_lst
    assert data_processor.file_options_dict == file_options_dict
    assert data_processor.file_name_identifier == file_name_identifier


def test_get_raw_data_from_file(
    default_data_processor_npy,
    data_path,
    dummy_data,
    write_dummy_data,
):
    """Test get raw data from file."""
    raw_data = default_data_processor_npy._get_raw_data_from_file(data_path)
    np.testing.assert_array_equal(raw_data, dummy_data)


def test_non_existing_file(default_data_processor_npy, data_path):
    """Test non-existing raw data file."""
    default_data_processor_npy.file_path = Path("non_existing_file.npy")
    raw_data = default_data_processor_npy._get_raw_data_from_file(data_path)
    assert raw_data is None


def test_wrong_file_type(default_data_processor_npy, write_wrong_dummy_data, wrong_data_path):
    """Test with wrong file type."""
    raw_data = default_data_processor_npy._get_raw_data_from_file(wrong_data_path)
    assert raw_data is None


def test_filter_and_manipulate_raw_data(default_data_processor_npy, dummy_data):
    """Test filter and manipulate raw data."""
    processed_data = default_data_processor_npy._filter_and_manipulate_raw_data(dummy_data)
    np.testing.assert_array_equal(processed_data, dummy_data)
