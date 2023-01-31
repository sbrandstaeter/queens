"""A module that gathers functionalities for output writing."""

import csv
import pathlib


def write_to_csv(output_file_path, data, delimiter=","):
    """A very simple csv file writer.

    Write data out to a csv-file. Nothing fancy, at the moment,
    only now header line or index column is supported just pure data.

    Args:
        output_file_path (Path obj): Path to the file the data should be written to
        data (np.array): Data in form of numpy arrays
        delimiter (optional, str): Delimiter to separate individual data.
                                   Defaults to comma delimiter.
    """
    # Write data to new file
    with open(pathlib.Path(output_file_path), 'w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file, delimiter=delimiter)
        # write only new data
        for row in data:
            writer.writerow(row)
