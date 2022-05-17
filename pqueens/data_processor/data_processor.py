"""Module for data processing of simulation results."""

import abc
import glob
import logging
import os

import numpy as np

_logger = logging.getLogger(__name__)


class DataProcessor(metaclass=abc.ABCMeta):
    """Base class for data processing.

    Attributes:
        file_name_identifier (str): Identifier for files.
                                         The file prefix can contain BASIC regex expression and
                                         subdirectories. Examples are wildcards `*` or
                                         expressions like `[ab]`.
        file_path (str): Actual path to the file of interest
        file_options_dict (dict): Dictionary with read-in options for
                                the file
        files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                The paths can contain regex expressions.
        driver_name (str): Name of the associated driver.
        raw_file_data (np.array): Raw data from file.
        processed_data (np.array): Cleaned, filtered or manipulated data_processor data
    """

    def __init__(
        self,
        file_name_identifier,
        file_options_dict,
        files_to_be_deleted_regex_lst,
        driver_name,
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
            driver_name (str): Name of the associated driver.
        """
        self.files_to_be_deleted_regex_lst = files_to_be_deleted_regex_lst
        self.file_options_dict = file_options_dict
        self.driver_name = driver_name
        self.file_name_identifier = file_name_identifier
        self.file_path = None
        self.processed_data = np.empty(shape=0)
        self.raw_file_data = None

    @classmethod
    def from_config_set_base_attributes(cls, config, driver_name):
        """Extract attributes of this base class from the config.

        Args:
            config (dict): Dictionary with problem description.
            driver_name (str): Name of driver that is used in this job-submission

        Returns:
            file_name_identifier (str): Identifier for files.
                                             The file prefix can contain regex expression and
                                             subdirectories.
            file_options_dict (dict): Dictionary with read-in options for
                                      the file. The respective child class will
                                       implement valid options for this dictionary.
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
            driver_name (str): Name of the associated driver.
        """
        driver_params = config.get(driver_name)
<<<<<<< HEAD:pqueens/post_post/post_post.py
        try:
            post_post_options = driver_params["driver_params"].get('post_post')
        except KeyError:
            post_post_options = driver_params.get("post_post")
=======
        data_processor_options = driver_params["driver_params"].get('data_processor')
>>>>>>> master:pqueens/data_processor/data_processor.py

        file_name_identifier = data_processor_options.get('file_name_identifier')
        if not file_name_identifier:
            raise IOError(
                f"No option 'file_name_identifier' was provided in '{driver_name}' driver! "
                "DataProcessor object cannot be instantiated! Abort..."
            )
        if not isinstance(file_name_identifier, str):
            raise TypeError(
                "The option 'file_name_identifier' must be of type 'str' "
                f"but is of type {type(file_name_identifier)}. Abort..."
            )

        file_options_dict = data_processor_options.get('file_options_dict')
        if not file_options_dict:
            raise IOError(
                f"No option 'file_options_dict' was provided in {driver_name} driver! "
                "DataProcessor object cannot be instantiated! Abort..."
            )
        if not isinstance(file_options_dict, dict):
            raise TypeError(
                "The option 'file_options_dict' must be of type 'dict' "
                f"but is of type {type(file_options_dict)}. Abort..."
            )

        files_to_be_deleted_regex_lst = data_processor_options.get(
            'files_to_be_deleted_regex_lst', []
        )
        if not isinstance(files_to_be_deleted_regex_lst, list):
            raise TypeError(
                "The option 'files_to_be_deleted_regex_lst' must be of type 'list' "
                f"but is of type {type(files_to_be_deleted_regex_lst)}. Abort..."
            )

        return file_name_identifier, file_options_dict, files_to_be_deleted_regex_lst

    def get_data_from_file(self, base_dir_file):
        """Get data of interest from file.

        Args:
            base_dir_file (str): Path of the base directory that
                                           contains the file of interest.

        Returns:
            processed_data (np.array): Final data from data processor module
        """
        if not base_dir_file:
            raise ValueError(
                "The data processor requires a base_directory for the "
                "files to operate on! Your input was empty! Abort..."
            )
        if not isinstance(base_dir_file, str):
            raise TypeError(
                "The argument 'base_dir_file' must be of type 'str' "
                f"but is of type {type(base_dir_file)}. Abort..."
            )

<<<<<<< HEAD:pqueens/post_post/post_post.py
        post_file_path_regex = self._generate_path_to_post_file(base_dir_post_file)
        file_exists_bool = self._check_file_exist_and_is_unique(post_file_path_regex)
        breakpoint()
=======
        file_path_regex = self._generate_path_to_file(base_dir_file)
        file_exists_bool = self._check_file_exist_and_is_unique(file_path_regex)
>>>>>>> master:pqueens/data_processor/data_processor.py
        if file_exists_bool:
            breakpoint()
            self._get_raw_data_from_file()
            self._filter_and_manipulate_raw_data()
            self._subsequent_data_manipulation()

        for file_path in self.files_to_be_deleted_regex_lst:
            self._clean_up(file_path)

        return self.processed_data

    def _generate_path_to_file(self, base_dir_file):
        """Generate path to file.

        Args:
            base_dir_file (str): Path to base directory that contains file of interest

        Returns:
            file_path_regex (str): Path to file that still
                                        contains wildcards or regex expressions
        """
<<<<<<< HEAD:pqueens/post_post/post_post.py
        file_identifier = self.post_file_name_identifier
        post_file_path_regex = os.path.join(base_dir_post_file, file_identifier)
        return post_file_path_regex
=======
        file_identifier = self.file_name_identifier + '*'
        file_path_regex = os.path.join(base_dir_file, file_identifier)
        return file_path_regex
>>>>>>> master:pqueens/data_processor/data_processor.py

    def _check_file_exist_and_is_unique(self, file_path_regex):
        """Check if file exists.

        Args:
            file_path_regex (str): Path to file that still
                                        contains wildcards or regex expressions

        Returns:
            file_exists (bool): Boolean determine whether file exists. If true, the file
                                exists.
        """
        file_list = glob.glob(file_path_regex)
        if len(file_list) > 1:
            raise RuntimeError(
                "The data_processor module found several files for the "
                "provided 'file_name_prefix'!"
                "The files are: {file_list}."
                "The file prefix must lead to a unique file. Abort..."
            )
<<<<<<< HEAD:pqueens/post_post/post_post.py
        if len(file_list) == 1:
            self.post_file_path = file_list[0]
=======
        elif len(file_list) == 1:
            self.file_path = file_list[0]
>>>>>>> master:pqueens/data_processor/data_processor.py
            file_exists = True
        else:
            _logger.warning(f"The file '{file_path_regex}' does not exist!")
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

    @staticmethod
    def _clean_up(file_path):
        """Clean-up files in the output directory.

        Args:
            file_path (str): File path with optional regex for the file that
                             should be deleted.

        Returns:
            None
        """
        try:
            os.remove(file_path)
        except (OSError, FileNotFoundError) as exception:
            _logger.debug(
                f"Could not remove file with path: '{file_path}'. "
                f"The following error was raised: {exception}"
            )
