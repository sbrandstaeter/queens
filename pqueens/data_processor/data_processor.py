"""Module for data processing of simulation results."""

import abc
import logging
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)


class DataProcessor(metaclass=abc.ABCMeta):
    """Base class for data processing.

    Attributes:
        files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                             The paths can contain regex expressions. The
                                             paths are relative to the particular simulation output
                                             folder: *experiment_dir/<job_id>/output/<here
                                             comes your regex>* .
        file_options_dict (dict): Dictionary with read-in options for
                                  the file.
        file_name_identifier (str): Identifier for files.
                                    The file prefix can contain BASIC regex expression
                                    and subdirectories. Examples are wildcards `*` or
                                    expressions like `[ab]`.
        file_path (str): Actual path to the file of interest.
        processed_data (np.array): Cleaned, filtered or manipulated *data_processor* data.
        raw_file_data (np.array): Raw data from file.
    """

    def __init__(
        self,
        file_name_identifier=None,
        file_options_dict=None,
        files_to_be_deleted_regex_lst=None,
    ):
        """Init data processor class.

        Args:
            file_name_identifier (str): Identifier for files.
                                             The file prefix can contain regex expression and
                                            subdirectories.
            file_options_dict (dict): Dictionary with read-in options for
                                      the file. The respective child class will
                                       implement valid options for this dictionary.
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
        """
        if not file_name_identifier:
            raise IOError(
                f"No option 'file_name_identifier' was provided in '{self.__class__.__name__}'! "
                "DataProcessor object cannot be instantiated! Abort..."
            )
        if not isinstance(file_name_identifier, str):
            raise TypeError(
                "The option 'file_name_identifier' must be of type 'str' "
                f"but is of type {type(file_name_identifier)}. Abort..."
            )

        if file_options_dict is None:
            raise IOError(
                f"No option 'file_options_dict' was provided in '{self.__class__.__name__}'! "
                "DataProcessor object cannot be instantiated! Abort..."
            )
        if not isinstance(file_options_dict, dict):
            raise TypeError(
                "The option 'file_options_dict' must be of type 'dict' "
                f"but is of type {type(file_options_dict)}. Abort..."
            )

        if files_to_be_deleted_regex_lst is None:
            files_to_be_deleted_regex_lst = []
        if not isinstance(files_to_be_deleted_regex_lst, list):
            raise TypeError(
                "The option 'files_to_be_deleted_regex_lst' must be of type 'list' "
                f"but is of type {type(files_to_be_deleted_regex_lst)}. Abort..."
            )

        self.files_to_be_deleted_regex_lst = files_to_be_deleted_regex_lst
        self.file_options_dict = file_options_dict
        self.file_name_identifier = file_name_identifier
        self.file_path = None
        self.processed_data = np.empty(shape=0)
        self.raw_file_data = None

    def get_data_from_file(self, base_dir_file):
        """Get data of interest from file.

        Args:
            base_dir_file (Path): Path of the base directory that
                                           contains the file of interest

        Returns:
            processed_data (np.array): Final data from data processor module
        """
        if not base_dir_file:
            raise ValueError(
                "The data processor requires a base_directory for the "
                "files to operate on! Your input was empty! Abort..."
            )
        if not isinstance(base_dir_file, Path):
            raise TypeError(
                "The argument 'base_dir_file' must be of type 'Path' "
                f"but is of type {type(base_dir_file)}. Abort..."
            )

        file_exists_bool = self._check_file_exist_and_is_unique(base_dir_file)
        if file_exists_bool:
            self._get_raw_data_from_file()
            self._filter_and_manipulate_raw_data()
            self._subsequent_data_manipulation()

        self._clean_up(base_dir_file)
        return self.processed_data

    def _check_file_exist_and_is_unique(self, base_dir_file):
        """Check if file exists.

        Args:
            base_dir_file (Path): Path to base directory that contains file of interest

        Returns:
            file_exists (bool): Boolean determine whether file exists. If true, the file
                            exists.
        """
        file_list = list(base_dir_file.glob(self.file_name_identifier))

        if len(file_list) > 1:
            raise RuntimeError(
                "The data_processor module found several files for the "
                "provided 'file_name_prefix'!"
                "The files are: {file_list}."
                "The file prefix must lead to a unique file. Abort..."
            )
        if len(file_list) == 1:
            self.file_path = file_list[0]
            file_exists = True
        else:
            _logger.warning(
                "The file '%s' does not exist!", base_dir_file / self.file_name_identifier
            )
            file_exists = False

        return file_exists

    @abc.abstractmethod
    def _get_raw_data_from_file(self):
        """Get the raw data from the files of interest."""

    @abc.abstractmethod
    def _filter_and_manipulate_raw_data(self):
        """Filter or clean the raw data for given criteria."""

    def _subsequent_data_manipulation(self):
        """Subsequent manipulate the data_processor data.

        This method can be easily implemented by overloading the empty
        method in a custom inheritance of a desired child class. Make
        sure to add the module to the `from_config_create` method in
        this file.
        """

    def _clean_up(self, base_dir_file):
        """Clean-up files in the output directory.

        Args:
            base_dir_file (Path): Path of the base directory that
                                    contains the file of interest.
        """
        current_file = None
        for regex in self.files_to_be_deleted_regex_lst:
            for file in sorted(base_dir_file.glob(regex)):
                current_file = file
                file.unlink(missing_ok=True)
