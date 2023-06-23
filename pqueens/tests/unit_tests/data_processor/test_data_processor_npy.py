"""Tests for numpy data processor."""

from pathlib import Path

import numpy as np
import pytest

from pqueens.data_processor.data_processor_numpy import DataProcessorNumpy


# ------ fixtures ----------
@pytest.fixture()
def data_path(tmp_path):
    """Create a dummy data path."""
    my_data_path = tmp_path / "my_data.npy"
    return my_data_path


@pytest.fixture()
def wrong_data_path(tmp_path):
    """Create a dummy data path."""
    my_data_path = tmp_path / "my_data.csv"
    return my_data_path


@pytest.fixture()
def dummy_data():
    """Create some dummy data."""
    data = np.array([[1, 2], [3, 4]])
    return data


@pytest.fixture()
def write_dummy_data(data_path, dummy_data):
    """Write the dummy data."""
    with open(data_path, "wb") as f:
        np.save(f, dummy_data)


@pytest.fixture()
def write_wrong_dummy_data(wrong_data_path, dummy_data):
    """Write the wrong dummy data."""
    with open(wrong_data_path, "w") as f:
        np.savetxt(f, dummy_data)


@pytest.fixture()
def default_data_processor_npy(data_path):
    """Dummy data processor npy."""
    file_name_identifier = "dummy"
    file_options_dict = {}
    files_to_be_deleted_regex_lst = []
    data_processor_name = 'npy_data_processor'

    data_processor = DataProcessorNumpy(
        data_processor_name,
        file_name_identifier,
        file_options_dict,
        files_to_be_deleted_regex_lst,
    )
    data_processor.file_path = data_path

    return data_processor


# ------ actual tests ------
def test_init():
    """Test initialization of npy data processor."""
    file_name_identifier = 'dummy'
    file_options_dict = {"dummy": "dummy"}
    files_to_be_deleted_regex_lst = ['abc']
    data_processor_name = 'npy_data_processor'

    data_processor = DataProcessorNumpy(
        data_processor_name,
        file_name_identifier,
        file_options_dict,
        files_to_be_deleted_regex_lst,
    )

    assert data_processor.files_to_be_deleted_regex_lst == files_to_be_deleted_regex_lst
    assert data_processor.file_options_dict == file_options_dict
    assert data_processor.data_processor_name == data_processor_name
    assert data_processor.file_name_identifier == file_name_identifier
    assert data_processor.file_path is None
    np.testing.assert_array_equal(data_processor.processed_data, np.empty(shape=0))
    assert data_processor.raw_file_data is None


def test_get_raw_data_from_file(
    default_data_processor_npy,
    dummy_data,
    write_dummy_data,
):
    """Test get raw data from file."""
    default_data_processor_npy._get_raw_data_from_file()
    np.testing.assert_array_equal(default_data_processor_npy.raw_file_data, dummy_data)


def test_non_existing_file(
    default_data_processor_npy,
):
    """Test non-existing raw data file."""
    default_data_processor_npy.file_path = Path("non_existing_file.npy")
    default_data_processor_npy._get_raw_data_from_file()
    assert default_data_processor_npy.raw_file_data is None


def test_wrong_file_type(default_data_processor_npy, write_wrong_dummy_data, wrong_data_path):
    """Test with wrong file type."""
    default_data_processor_npy.file_path = wrong_data_path
    default_data_processor_npy._get_raw_data_from_file()
    assert default_data_processor_npy.raw_file_data is None


def test_filter_and_manipulate_raw_data(default_data_processor_npy, dummy_data):
    """Test filter and manipulate raw data."""
    default_data_processor_npy.raw_file_data = dummy_data
    default_data_processor_npy._filter_and_manipulate_raw_data()
    np.testing.assert_array_equal(default_data_processor_npy.processed_data, dummy_data)
