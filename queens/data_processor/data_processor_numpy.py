"""Data processor class for csv data extraction."""

import logging

import numpy as np

from queens.data_processor.data_processor import DataProcessor
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class DataProcessorNumpy(DataProcessor):
    """Class for extracting data from numpy binaries."""

    @log_init_args
    def __init__(
        self,
        file_name_identifier=None,
        file_options_dict=None,
        files_to_be_deleted_regex_lst=None,
    ):
        """Instantiate data processor class for numpy binary data.

        Args:
            file_name_identifier (str): Identifier of file name.
                                        The file prefix can contain regex expression
                                        and subdirectories.
            file_options_dict (dict): Dictionary with read-in options for the file
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.

        Returns:
            Instance of DataProcessorNpy class
        """
        super().__init__(
            file_name_identifier=file_name_identifier,
            file_options_dict=file_options_dict,
            files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
        )

    def _get_raw_data_from_file(self, file_path):
        """Get the raw data from the files of interest.

        This method loads the numpy binary data from the file.

        Args:
            file_path (str): Actual path to the file of interest.

        Returns:
            raw_data (np.array): Raw data from file.
        """
        try:
            raw_data = np.load(file_path)
            _logger.info("Successfully read-in data from %s.", file_path)
            return raw_data
        except FileNotFoundError as error:
            _logger.warning(
                "Could not find the file: %s. The following FileNotFoundError was raised: %s. "
                "Skipping the file and continuing.",
                file_path,
                error,
            )
        except ValueError as error:
            _logger.warning(
                "Could not read the file: %s. The following ValueError was raised: %s. "
                "Skipping the file and continuing.",
                file_path,
                error,
            )
        return None

    def _filter_and_manipulate_raw_data(self, raw_data):
        """Filter and manipulate the raw data.

        In this case we want the raw data as it is. Of course this
        method can implement more specific data processing in the
        future.

        Args:
            raw_data (np.array): Raw data from file.

        Returns:
            processed_data (np.array): Cleaned, filtered or manipulated *data_processor* data.
        """
        return raw_data
