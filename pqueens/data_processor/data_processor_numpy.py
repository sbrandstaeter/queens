"""Data processor class for csv data extraction."""

import logging

import numpy as np

from pqueens.data_processor.data_processor import DataProcessor

_logger = logging.getLogger(__name__)


class DataProcessorNumpy(DataProcessor):
    """Class for extracting data from numpy binaries."""

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

    def _get_raw_data_from_file(self):
        """Get the raw data from the files of interest.

        This method loads the numpy binary data from the file.
        """
        try:
            self.raw_file_data = np.load(self.file_path)
            _logger.info("Successfully read-in data from %s.", self.file_path)
        except FileNotFoundError as error:
            _logger.warning(
                "Could not find file %s. The FileNotFoundError was: %s. Skip...",
                self.file_path,
                error,
            )
            self.raw_file_data = None
        except ValueError as error:
            _logger.warning(
                "Could not read file %s. The ValueError was: %s. Skip...", self.file_path, error
            )
            self.raw_file_data = None

    def _filter_and_manipulate_raw_data(self):
        """Filter and manipulate the raw data.

        In this case we want the raw data as it is. Of course this
        method can implement more specific data processing in the
        future.
        """
        self.processed_data = self.raw_file_data
