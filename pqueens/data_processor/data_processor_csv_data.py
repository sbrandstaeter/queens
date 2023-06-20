"""Data processor class for csv data extraction."""

import logging

import numpy as np
import pandas as pd

from pqueens.data_processor.data_processor import DataProcessor
from pqueens.utils.valid_options_utils import get_option

_logger = logging.getLogger(__name__)


class DataProcessorCsv(DataProcessor):
    """Class for extracting data from csv files.

    Attributes:
        use_cols_lst (lst): List with column numbers that should be read-in.
        filter_type (str): Filter type to use.
        header_row (int): Integer that determines which csv-row contains labels/headers of
                          the columns. Default is 'None', meaning no header used.
        skip_rows (int): Number of rows that should be skipped to be read-in in csv file.
        index_column (int, str): Column to use as the row labels of the DataFrame, either
                                 given as string name or column index.

                                 **Note:** *index_column=False* can be used to force pandas
                                 to not use the first column as the index. *index_column* is
                                 used for filtering the remaining columns.
        use_rows_lst (lst): In case this options is used, the list contains the indices of rows
                            in the csv file that should be used as data.
        filter_range (lst): After data is selected by *use_cols_lst* and a filter column is
                            specified by *index_column*, this option selects which data range
                            shall be filtered by providing a minimum and maximum value pair
                            in list format.
        filter_target_values (list): Target values to filter.
        filter_tol (float): Tolerance for the filter range.
        returned_filter_format (str): Returned data format after filtering.
    """

    expected_filter_entire_file = {'type': 'entire_file'}
    expected_filter_by_row_index = {'type': 'by_row_index', 'rows': [1, 2]}
    expected_filter_by_target_values = {
        'type': 'by_target_values',
        'target_values': [1.0, 2.0, 3.0],
        'tolerance': 0.0,
    }
    expected_filter_by_range = {'type': 'by_range', 'range': [1.0, 2.0], 'tolerance': 0.0}

    def __init__(
        self,
        data_processor_name,
        file_name_identifier=None,
        file_options_dict=None,
        files_to_be_deleted_regex_lst=None,
    ):
        """Instantiate data processor class for csv data.

        Args:
            data_processor_name (str): Name of the data processor.
            file_name_identifier (str): Identifier of file name
                                             The file prefix can contain regex expression
                                             and subdirectories.
            file_options_dict (dict): Dictionary with read-in options for the file:
                - header_row (int): Integer that determines which csv-row contains labels/headers of
                                    the columns. Default is 'None', meaning no header used.
                - use_cols_lst (lst): (optional) list with column numbers that should be read-in.
                - skip_rows (int): Number of rows that should be skipped to be read-in in csv file.
                - index_column (int, str): Column to use as the row labels of the DataFrame, either
                                           given as string name or column index.
                                           Note: index_column=False can be used to force pandas to
                                           not use the first column as the index. Index_column is
                                           used for filtering the remaining columns.
                - returned_filter_format (str): Returned data format after filtering
                - filter (dict): Dictionary with filter options:
                    -- type (str): Filter type to use
                    -- rows (lst): In case this options is used, the list contains the indices of
                                  rows in the csv file that should be used as data
                    -- range (lst): After data is selected by `use_cols_lst` and a filter column
                                   is specified by `index_column`, this option selects which data
                                   range shall be filtered by providing a minimum and maximum
                                   value pair in list format
                    -- target_values (list): target values to filter
                    -- tolerance (float): Tolerance for the filter range

            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.

        Returns:
            Instance of DataProcessorCsv class
        """
        super().__init__(
            data_processor_name=data_processor_name,
            file_name_identifier=file_name_identifier,
            file_options_dict=file_options_dict,
            files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
        )

        header_row = file_options_dict.get('header_row')
        if header_row and not isinstance(header_row, int):
            raise ValueError(
                "The option 'header_row' in the data_processor settings must be of type 'int'! "
                f"You provided type '{type(header_row)}'. Abort..."
            )

        use_cols_lst = file_options_dict.get('use_cols_lst')
        if use_cols_lst and not isinstance(use_cols_lst, list):
            raise TypeError(
                "The option 'use_cols_lst' must be of type 'list' "
                f"but you provided type {type(use_cols_lst)}. Abort..."
            )

        skip_rows = file_options_dict.get('skip_rows', 0)
        if not isinstance(skip_rows, int):
            raise ValueError(
                "The option 'skip_rows' in the data_processor settings must be of type 'int'! "
                f"You provided type '{type(skip_rows)}'. Abort..."
            )

        index_column = file_options_dict.get('index_column', False)
        if index_column and not isinstance(index_column, (int, str)):
            raise TypeError(
                "The option 'index_column' must be either of type 'int' or 'str', "
                f"but you provided type {type(index_column)}! Either your original data "
                "type is wrong or the column does not exist in the csv-data file! "
                "Abort..."
            )

        returned_filter_format = file_options_dict.get('returned_filter_format', 'numpy')

        filter_options_dict = file_options_dict.get('filter')
        self._check_valid_filter_options(filter_options_dict)

        filter_type = filter_options_dict.get('type')
        if not isinstance(filter_type, str):
            raise ValueError(
                "The option 'filter_type' in the data_processor settings must be of type 'str'! "
                f"You provided type '{type(filter_type)}'. Abort..."
            )

        use_rows_lst = filter_options_dict.get('rows', [])
        if not isinstance(use_rows_lst, list):
            raise TypeError(
                "The option 'use_rows_lst' must be of type 'list' "
                f"but you provided type {type(use_rows_lst)}. Abort..."
            )
        if not all(isinstance(row_idx, int) for row_idx in use_rows_lst):
            raise TypeError(
                "The option 'use_rows_lst' must be a list of `int` "
                f"but you provided type {[type(row_idx) for row_idx in use_rows_lst]}. Abort..."
            )

        filter_range = filter_options_dict.get('range', [])
        if filter_range and not isinstance(filter_range, list):
            raise TypeError(
                "The option 'filter_range' has to be of type 'list', "
                f"but you provided type {type(filter_range)}. Abort..."
            )

        filter_target_values = filter_options_dict.get('target_values', [])
        if not isinstance(filter_target_values, list):
            raise TypeError(
                "The option 'target_values' has to be of type 'list', "
                f"but you provided type {type(filter_target_values)}. Abort..."
            )

        filter_tol = filter_options_dict.get('tolerance', 0.0)
        if not isinstance(filter_tol, float):
            raise TypeError(
                "The option 'filter_tol' has to be of type 'float', "
                f"but you provided type {type(filter_tol)}. Abort..."
            )

        self.use_cols_lst = use_cols_lst
        self.filter_type = filter_type
        self.header_row = header_row
        self.skip_rows = skip_rows
        self.index_column = index_column
        self.use_rows_lst = use_rows_lst
        self.filter_range = filter_range
        self.filter_target_values = filter_target_values
        self.filter_tol = filter_tol
        self.returned_filter_format = returned_filter_format

    @classmethod
    def _check_valid_filter_options(cls, filter_options_dict):
        """Check valid filter input options.

        Args:
            filter_options_dict (dict): dictionary with filter options
        """
        if filter_options_dict["type"] == 'entire_file':
            if not filter_options_dict.keys() == cls.expected_filter_entire_file.keys():
                raise TypeError(
                    "For the filter type `entire_file`, you have to provide "
                    f"a dictionary of type {cls.expected_filter_entire_file}."
                )
            return
        if filter_options_dict["type"] == 'by_range':
            if not filter_options_dict.keys() == cls.expected_filter_by_range.keys():
                raise TypeError(
                    "For the filter type `by_range`, you have to provide "
                    f"a dictionary of type {cls.expected_filter_by_range}."
                )
            return
        if filter_options_dict["type"] == 'by_row_index':
            if not filter_options_dict.keys() == cls.expected_filter_by_row_index.keys():
                raise TypeError(
                    "For the filter type `by_row_index`, you have to provide "
                    f"a dictionary of type {cls.expected_filter_by_row_index}."
                )
            return
        if filter_options_dict["type"] == 'by_target_values':
            if not filter_options_dict.keys() == cls.expected_filter_by_target_values.keys():
                raise TypeError(
                    "For the filter type `by_target_values`, you have to provide "
                    f"a dictionary of type {cls.expected_filter_by_target_values}."
                )
        else:
            raise TypeError("You provided an invalid 'filter_type'!")

    def _get_raw_data_from_file(self):
        """Get the raw data from the files of interest.

        This method loads the desired parts of the csv file as a pandas
        dataframe.
        """
        try:
            self.raw_file_data = pd.read_csv(
                self.file_path,
                sep=r',|\s+',
                usecols=self.use_cols_lst,
                skiprows=self.skip_rows,
                header=self.header_row,
                engine='python',
                index_col=self.index_column,
            )
            _logger.info("Successfully read-in data from %s.", self.file_path)
        except IOError as error:
            _logger.warning(
                "Could not read file %s. The IOError was: %s. Skip...", self.file_path, error
            )
            self.raw_file_data = None

    def _filter_and_manipulate_raw_data(self):
        """Filter the pandas data-frame based on filter type."""
        valid_filter_types = {
            'entire_file': self._filter_entire_file,
            'by_range': self._filter_by_range,
            'by_row_index': self._filter_by_row_index,
            'by_target_values': self._filter_by_target_values,
        }

        error_message = "You provided an invalid 'filter_type'!"
        filter_method = get_option(
            valid_filter_types, self.filter_type, error_message=error_message
        )
        filter_method()
        filter_formats_dict = {
            "numpy": self.processed_data.to_numpy(),
            "dict": self.processed_data.to_dict('list'),
        }

        self.processed_data = get_option(
            filter_formats_dict,
            self.returned_filter_format,
            error_message="The returned filter format you provided is not a current option.",
        )

        if not np.any(self.processed_data):
            raise RuntimeError(
                "The filtered data was empty! Adjust your filter tolerance or filter range!"
            )

    def _filter_entire_file(self):
        """Keep entire csv file data."""
        self.processed_data = self.raw_file_data

    def _filter_by_row_index(self):
        """Filter the csv file based on given data rows."""
        if any(self.raw_file_data):
            try:
                self.processed_data = self.raw_file_data.iloc[self.use_rows_lst]
            except IndexError as exception:
                raise IndexError(
                    f"Index list {self.use_rows_lst} are not contained in raw_file_data. "
                ) from exception

    def _filter_by_target_values(self):
        """Filter the pandas data frame based on target values."""
        if any(self.raw_file_data):
            target_indices = []
            for target_value in self.filter_target_values:
                target_indices.append(
                    int(
                        np.where(
                            np.abs(self.raw_file_data.index - target_value) <= self.filter_tol
                        )[0]
                    )
                )

            self.processed_data = self.raw_file_data.iloc[target_indices]

    def _filter_by_range(self):
        """Filter the pandas data frame based on values in a data column."""
        if any(self.raw_file_data):
            range_start = int(
                np.where(
                    np.abs(self.raw_file_data.index - self.filter_range[0]) <= self.filter_tol
                )[0]
            )
            range_end = int(
                np.where(
                    np.abs(self.raw_file_data.index - self.filter_range[-1]) <= self.filter_tol
                )[-1]
            )

            self.processed_data = self.raw_file_data.iloc[range_start : range_end + 1]
