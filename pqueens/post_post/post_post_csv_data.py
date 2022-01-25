"""Post post class for general csv data extraction."""

import logging
from email import header
from time import time

import pandas as pd

from pqueens.post_post.post_post import PostPost

_logger = logging.getLogger(__name__)


class PostPostCsv(PostPost):
    """Class for extracting data from csv files."""

    def __init__(
        self,
        post_post_files_regex_lst,
        file_options_lst,
        files_to_be_deleted_regex_lst,
        driver_name,
    ):
        """Instantiate post post class for csv data.

        Args:
            post_post_files_regex_lst (lst): List with paths to postprocessed files.
                                             The file paths can contain regex expression.
            file_options_lst (lst): List containing dictionaries with read-in options for
                                    the post_processed files
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
            driver_name (str): Name of the associated driver.

        Returns:
            Instance of PostPostCsv class
        """
        super(PostPostCsv, self).__init__(
            post_post_files_regex_lst, file_options_lst, files_to_be_deleted_regex_lst, driver_name
        )

    @classmethod
    def from_config_create_post_post(cls, post_post_options, driver_name):
        """Create the class from the problem description.

        Args:
            post_post_options (dict): Dictionary containing the problem description
            driver_name (str): Name of the associated driver

        Returns:
            Instance of the PostPostCsv class
        """
        post_post_files_regex_lst = post_post_options.get('post_post_files_regex_lst')
        if not post_post_files_regex_lst:
            raise IOError(
                f"No option 'post_post_files_regex_lst' was provided in {driver_name} driver! "
                "PostPost object cannot be instantiated! Abort..."
            )
        if not isinstance(post_post_files_regex_lst, list):
            raise TypeError(
                "The option 'post_post_files_regex_lst' must be of type 'list' "
                f"but is of type {type(post_post_files_regex_lst)}. Abort..."
            )

        file_options_lst = post_post_options.get('file_options_lst')
        if not file_options_lst:
            raise IOError(
                f"No option 'file_options_lst' was provided in {driver_name} driver! "
                "PostPost object cannot be instantiated! Abort..."
            )
        if not isinstance(file_options_lst, list):
            raise TypeError(
                "The option 'file_options_lst' must be of type 'list' "
                f"but is of type {type(file_options_lst)}. Abort..."
            )

        files_to_be_deleted_regex_lst = post_post_options.get('files_to_be_deleted_regex_lst', [])
        if not isinstance(files_to_be_deleted_regex_lst, list):
            raise TypeError(
                "The option 'files_to_be_deleted_regex_lst' must be of type 'list' "
                f"but is of type {type(files_to_be_deleted_regex_lst)}. Abort..."
            )

        return cls(
            post_post_files_regex_lst,
            file_options_lst,
            files_to_be_deleted_regex_lst,
            driver_name,
        )

    def _get_raw_data_from_file(self, file_path, file_options_dict):
        """Get the raw data from the files of interest.

        Args:
            file_path (str): File path that can also contain regex
                             expressions.
            file_options_dict (dic): Dictionary with read-in options for postprocessed
                                     file of interest.

        Returns:
            None
        """
        csv_reader = self._get_csv_reader_from_options(file_options_dict)
        csv_reader(file_path, file_options_dict)

    def _get_csv_reader_from_options(self, file_options_dict):
        """Get the correct csv reader configuration.

        Based on the file_options_dict, decide which csv reader to return.
        Also check the file_options_dict for valid key-value combinations.

        Args:
            file_options_dict (dic): Dictionary with read-in options for postprocessed
                                     file of interest.

        Returns:
            csv_reader (obj): Correct csv-reader method.
        """
        option_keys = list(file_options_dict.keys())

        if ('use_rows_lst' in option_keys) and not ('time_range' in option_keys):
            csv_reader = self._read_csv_given_row_range
        elif ('time_range' in option_keys) and not ('use_rows_lst' in option_keys):
            csv_reader = self._read_csv_given_time_range
        else:
            raise ValueError("")

        return csv_reader

    def _read_csv_given_time_range(self, file_path, file_options_dict):
        """Read in csv data based on predefined time range.

        Args:
            file_path (str): Path to file of interest. Might contain regex experssions.
            file_options_dict (dict): Dictionary with options how to read the data.

        Returns:
            filtered_file_data (np.array): Numpy array with filtered data
        """
        use_cols_lst = file_options_dict.get('use_cols_lst')
        if not use_cols_lst:
            raise ValueError()

        time_tol = file_options_dict.get('time_tol')
        if not time_tol:
            raise ValueError()

        time_range = file_options_dict.get('time_range')
        if not time_range:
            raise ValueError()

        header = file_options_dict.get('header')
        skip_rows_lst = file_options_dict('skip_rows_lst')

        try:
            file_data = pd.read_csv(
                file_path,
                sep=r',|\s+',
                usecols=use_cols_lst,
                skiprows=skip_rows_lst,
                header=header,
                engine='python',
            )
            _logger.info(f"Successfully read-in data from {file_path}.")
        except IOError:
            _logger.warning(f"Could not read postprocessed file {file_path}. Skip...")
            file_data = None

        filtered_file_data = PostPostCsv._filter_data_for_time_range(
            time_tol, time_range, file_data
        )

        return filtered_file_data

    @staticmethod
    def _filter_data_for_time_range(time_tol, time_range, file_data):
        """Filter the pandas data-frame for data in specific time range.

        Args:
            time_tol (float): Tolerance for matching the time value
            time_range (lst): List containing the target time, time range
                              or time indicator for filtering
            file_data (pd.dataframe): pandas data-frame containing time series.

        Returns:
            filtered_data (np.array): Numpy array with filtered data
        """
        pass

    def _read_csv_given_row_range(self, file_path, file_options_dict):
        """Read in the data from the csv-file based on a given row range.

        Args:
            file_path (str): Path to file of interest. Might contain regex experssions.
            file_options_dict (dict): Dictionary with options how to read the data.

        Returns:
            filtered_file_data (np.array): Numpy array with filtered data
        """
        # TODO continue here
        required_keys_list = [
            'header',
            'use_cols_lst',
            'time_tol',
            'time_range',
            'skip_rows_lst',
            'use_rows_lst',
        ]

        pass

    def _filter_and_manipulate_raw_data(self):
        """Manipulate the raw data."""
        pass
