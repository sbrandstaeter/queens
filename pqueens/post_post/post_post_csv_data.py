"""Post post class for csv data extraction."""

import logging

import numpy as np
import pandas as pd

from pqueens.post_post.post_post import PostPost

_logger = logging.getLogger(__name__)


class PostPostCsv(PostPost):
    """Class for extracting data from csv files.

    Attributes:
        header_row (int):   Interger that determines which csv-row contains labels/headers of the
                            columns. Default is 'None', meaning no header used.
        use_cols_lst (lst): (optional) list with column numbers that should be read-in.
        skip_rows (int): Number of rows that should be skiped to be read-in in csv file.
        filter_column (int, str): After data is selected by `use_cols_lst`, filter_column
                                    is used to specify the column number or name, of this subset,
                                    which is used for filtering the remaining columns row-wise.
        use_rows_lst (lst): In case this options is used, the list contains the indices of rows
                            in the csv file that should be used as post post data
        filter_range (lst): After data is selected by `use_cols_lst` and a filter column is
                            specified by `filter_column`, this option selects which data range
                            shall be filtered by providing a minimum and maximum value pair
                            in list format
        filter_tol (float): Tolerance for the filter range
    """

    def __init__(
        self,
        post_file_name_prefix,
        file_options_dict,
        files_to_be_deleted_regex_lst,
        driver_name,
        header_row,
        use_cols_lst,
        skip_rows,
        filter_column,
        use_rows_lst,
        filter_range,
        filter_tol,
    ):
        """Instantiate post post class for csv data.

        Args:
            post_file_name_prefix (str): Prefix of postprocessed file name
                                         The file prefix can contain regex expression
                                         and subdirectories.
            file_options_dict (dict): Dictionary with read-in options for
                                      the post_processed file
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
            driver_name (str): Name of the associated driver.
            header_row (int):   Interger that determines which csv-row contains labels/headers of
                                the columns. Default is 'None', meaning no header used.
            use_cols_lst (lst): (optional) list with column numbers that should be read-in.
            skip_rows (int): Number of rows that should be skiped to be read-in in csv file.
            filter_column (int, str): After data is selected by `use_cols_lst`, filter_column
                                      is used to specify the column number or name, of this subset,
                                      which is used for filtering the remaining columns row-wise.
            use_rows_lst (lst): In case this options is used, the list contains the indices of rows
                                in the csv file that should be used as post post data
            filter_range (lst): After data is selected by `use_cols_lst` and a filter column is
                                specified by `filter_column`, this option selects which data range
                                shall be filtered by providing a minimum and maximum value pair
                                in list format
            filter_tol (float): Tolerance for the filter range

        Returns:
            Instance of PostPostCsv class
        """
        super(PostPostCsv, self).__init__(
            post_file_name_prefix,
            file_options_dict,
            files_to_be_deleted_regex_lst,
            driver_name,
        )
        self.use_cols_lst = use_cols_lst
        self.header_row = header_row
        self.skip_rows = skip_rows
        self.filter_column = filter_column
        self.use_rows_lst = use_rows_lst
        self.filter_range = filter_range
        self.filter_tol = filter_tol

    @classmethod
    def from_config_create_post_post(cls, config, driver_name):
        """Create the class from the problem description.

        Args:
            config (dict): Dictionary with problem description.
            driver_name (str): Name of driver that is used in this job-submission
        """
        (
            post_file_name_prefix,
            file_options_dict,
            files_to_be_deleted_regex_lst,
        ) = super().from_config_set_base_attributes(config, driver_name)

        header_row = file_options_dict.get('header_row')
        if header_row:
            if not isinstance(header_row, int):
                raise ValueError(
                    "The option 'header_row' in the post_post settings must be of type 'int'! "
                    "You provided type '{type(header_bool)}'. Abort..."
                )

        use_cols_lst = file_options_dict.get('use_cols_lst')
        if not use_cols_lst:
            raise ValueError("The option 'use_cols_lst' cannot be empty! Abort...")

        skip_rows = file_options_dict.get('skip_rows', 0)
        if not isinstance(skip_rows, int):
            raise ValueError(
                "The option 'skip_rows' in the post_post settings must be of type 'int'! "
                "You provided type '{type(skip_rows)}'. Abort..."
            )

        filter_column = file_options_dict.get('filter_column')
        if filter_column:
            if not (isinstance(filter_column, int) or isinstance(filter_column, str)):
                raise TypeError(
                    "The option 'filter_column' must be either of type 'int' or 'str', "
                    f"but you provided type {type(filter_column)}! Either your original data "
                    "type is wrong or the column does not exist in the csv-data file! "
                    "Abort..."
                )

        use_rows_lst = file_options_dict.get('use_rows_lst')
        if use_rows_lst:
            if not isinstance(use_rows_lst, list):
                raise TypeError(
                    "The option 'use_rows_lst' must be of type 'lst' "
                    "but you provided type {type(filter_column)}. Abort..."
                )

        filter_range = file_options_dict.get('filter_range')
        if filter_range:
            if not isinstance(filter_range, list):
                raise TypeError(
                    "The option 'filter_range' has to be of type 'list', "
                    f"but you provided type {type(filter_range)}. Abort..."
                )

        filter_tol = file_options_dict.get('filter_tol')
        if filter_tol:
            if not isinstance(filter_tol, float):
                raise TypeError(
                    "The option 'filter_tol' has to be of type 'float', "
                    f"but you provided type {type(filter_tol)}. Abort..."
                )

        return cls(
            post_file_name_prefix,
            file_options_dict,
            files_to_be_deleted_regex_lst,
            driver_name,
            header_row,
            use_cols_lst,
            skip_rows,
            filter_column,
            use_rows_lst,
            filter_range,
            filter_tol,
        )

    def _get_raw_data_from_file(self):
        """Get the raw data from the files of interest.

        This method loads the desired parts of the csv file as a pandas
        dataframe.
        """
        try:
            self.raw_file_data = pd.read_csv(
                self.post_file_path,
                sep=r',|\s+',
                usecols=self.use_cols_lst,
                skiprows=self.skip_rows,
                header=self.header_row,
                engine='python',
            )
            _logger.info(f"Successfully read-in data from {self.post_file_path}.")
        except IOError as error:
            _logger.warning(
                f"Could not read postprocessed file {self.post_file_path}. "
                f"The IOError was: {error}. Skip..."
            )
            self.raw_file_data = None

    def _filter_and_manipulate_raw_data(self):
        """Filter the pandas data-frame for data in specific time range."""
        if (self.use_rows_lst is not None) and (self.filter_column is None):
            self._filter_based_on_given_row_range()
        elif (self.filter_column is not None) and (self.use_rows_lst is None):
            self._filter_based_on_column_values()
        else:
            raise ValueError(
                "You provided a 'file_options_dict' with invalid keys! "
                "Valid keys must either contain 'use_rows_lst' but not 'filter_column' "
                f"or vice versa. Abort..."
            )

    def _filter_based_on_given_row_range(self):
        """Filter the csv file based on given data rows."""
        if any(self.raw_file_data):
            try:
                self.post_post_data = self.raw_file_data.iloc[self.use_rows_lst].to_numpy()
            except IndexError as exception:
                raise IndexError(
                    f"Index list {self.use_rows_lst} are not contained in raw_file_data. "
                    f"The IndexError was: {exception}. Abort..."
                )

    def _filter_based_on_column_values(self):
        """Filter the pandas data frame based on values in a data column."""
        if any(self.raw_file_data):
            post_post_data = (
                self.raw_file_data.loc[
                    (
                        np.abs(
                            self.raw_file_data.iloc[:, self.filter_column] - self.filter_range[0]
                        )
                        <= self.filter_tol
                    )
                    & (
                        np.abs(
                            self.raw_file_data.iloc[:, self.filter_column] - self.filter_range[-1]
                        )
                        <= self.filter_tol
                    )
                ]
            ).to_numpy()

            # delete the filter column from the data array
            self.post_post_data = np.delete(post_post_data, self.filter_column, axis=1)
            if not np.any(self.post_post_data):
                _logger.warning(
                    "The filtered data was empty! Adjust your filter tolerance or filter range!"
                )
