"""Unit tests for the output writer module."""

import pathlib

import numpy as np

from pqueens.utils.output_writer import write_to_csv


def test_write_to_csv(tmpdir):
    """Test csv writer."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # write out data
    output_file_path = pathlib.Path(tmpdir, "my_csv_file.csv")
    write_to_csv(output_file_path, data)

    # read data from written out file with basic readline routine
    read_in_data_lst = []
    with open(output_file_path, "r", encoding="utf-8") as in_file:
        for line in in_file:
            values = line.strip().split(',')
            read_in_data_lst.append(values)

    read_in_data = np.array(read_in_data_lst, dtype=np.float64)

    # read the data in again and compare to original data
    np.testing.assert_array_equal(data, read_in_data)
