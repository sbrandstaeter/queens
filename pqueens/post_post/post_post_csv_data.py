"""Post post class for general csv data extraction."""

import logging

import pandas as pd

from pqueens.post_post.post_post import PostPost

_logger = logging.getLogger(__name__)


class PostPostCsv(PostPost):
    """Class for extracting data from csv files."""

    def __init__(
        self,
        post_post_file_path,
        file_options_dict,
        files_to_be_deleted_regex_lst,
        driver_name,
    ):
        """Instantiate post post class for csv data.

        Args:
            post_post_file_path (str): Path to postprocessed files.
                                       The file path can contain regex expression.
            file_options_dict (dict): Dictionary with read-in options for
                                      the post_processed file
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
            driver_name (str): Name of the associated driver.

        Returns:
            Instance of PostPostCsv class
        """
        super(PostPostCsv, self).__init__(
            post_post_file_path, file_options_dict, files_to_be_deleted_regex_lst, driver_name
        )

    @classmethod
    def from_config_create_post_post(
        cls,
        driver_name,
        post_post_file_path,
        file_options_dict,
        files_to_be_deleted_regex_lst,
    ):
        """Create the class from the problem description."""
        return cls(
            post_post_file_path,
            file_options_dict,
            files_to_be_deleted_regex_lst,
            driver_name,
        )

    def _get_raw_data_from_file(self):
        """Get the raw data from the files of interest."""
        use_cols_lst = self.file_options_dict.get('use_cols_lst')
        if not use_cols_lst:
            raise ValueError("The option 'use_cols_lst' cannot be empty! Abort...")

        header = self.file_options_dict.get('header')
        skip_rows_lst = self.file_options_dict('skip_rows_lst')

        try:
            self.raw_file_data = pd.read_csv(
                self.post_post_file_path,
                sep=r',|\s+',
                usecols=use_cols_lst,
                skiprows=skip_rows_lst,
                header=header,
                engine='python',
            )
            _logger.info(f"Successfully read-in data from {self.post_post_file_path}.")
        except IOError:
            _logger.warning(f"Could not read postprocessed file {self.post_postfile_path}. Skip...")
            self.raw_file_data = None

    def _filter_raw_data(self):
        """Filter the pandas data-frame for data in specific time range."""
        option_keys = list(self.file_options_dict.keys())

        if ('use_rows_lst' in option_keys) and not ('filter_column' in option_keys):
            self._filter_based_on_given_row_range()
        elif ('filter_column' in option_keys) and not ('use_rows_lst' in option_keys):
            self._filter_based_on_column_values()
        else:
            raise ValueError(
                "You provided an 'file_options_dict' with invalid keys! "
                "Valid keys must either contain 'use_rows_lst' but not 'use_rows_lst' "
                f"or vice versa. You provided the keys: {option_keys}. Abort..."
            )

    def _filter_based_on_given_row_range(self):
        """Filter the csv file based on given data rows."""
        use_rows_lst = self.file_options_dict.get('use_rows_lst')
        if not use_rows_lst:
            raise ValueError(
                "The option 'use_rows_lst' was empty but a valid choice of rows "
                "is required to filter the csv file. Abort..."
            )
        if not isinstance(use_rows_lst, list):
            raise TypeError(
                "The option 'use_rows_lst' must be of type 'list', "
                f"but you provided type '{type(use_rows_lst)}'. Abort..."
            )

        if self.raw_file_data:
            try:
                self.post_post_data = self.raw_file_data.loc[use_rows_lst]
            except IndexError:
                raise IndexError(
                    f"Index list {use_rows_lst} are not contained in raw_file_data. " "Abort..."
                )

    def _filter_based_on_column_values(self):
        """Filter the pandas data fram based on values in a data column."""
        filter_tol = self.file_options_dict.get('filter_tol')
        if not filter_tol:
            raise ValueError("The option 'filter_tol' cannot be empty! Abort...")

        filter_range = self.file_options_dict.get('filter_range')
        if not filter_range:
            raise ValueError("The option 'filte_range' cannot be empty! Abort...")
        if not isinstance(filter_range, list):
            raise TypeError(
                "The option 'filter_range' has to be of type 'list', "
                f"but you provided type {type(filter_range)}. Abort..."
            )

        filter_column = self.file_options_dict.get('filter_column')
        if not (isinstance(filter_column, int) or isinstance(filter_column, str)):
            raise TypeError(
                "The option 'filter_column' must be either of type 'int' or 'str', "
                f"but you provided type {type(filter_column)}! Either your original data "
                "type is wrong or the column does not exist in the csv-data file! "
                "Abort..."
            )

        if self.raw_file_data:
            self.post_post_data = self.raw_file_data.loc[
                ((self.raw_file_data[filter_column] - filter_range[0]) > filter_tol)
                and ((self.raw_file_data[filter_column] - filter_range[-1]) < filter_tol)
            ]
            if not self.post_post_data:
                _logger.warning(
                    "The filtered data was empty! Adjust your filter tolerance or filter range!"
                )

    def _manipulate_data(self):
        """Manipulate the raw data."""
        pass
