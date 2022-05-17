"""Module for post-post processing of simulation results."""

import abc
import glob
import logging
import os

import numpy as np

_logger = logging.getLogger(__name__)


class PostPost(metaclass=abc.ABCMeta):
    """Base class for post post processing.

    Attributes:
        post_file_name_identifier (str): Identifier for postprocessed files.
                                         The file prefix can contain BASIC regex expression and
                                         subdirectories. Examples are wildcards `*` or
                                         expressions like `[ab]`.
        post_file_path (str): Actual path to the file of interest
        file_options_dict (dict): Dictionary with read-in options for
                                the post_processed file
        files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                The paths can contain regex expressions.
        driver_name (str): Name of the associated driver.
        raw_file_data (np.array): Raw data from file.
        post_post_data (np.array): Cleaned, filtered or manipulated post_post data
    """

    def __init__(
        self,
        post_file_name_identifier,
        file_options_dict,
        files_to_be_deleted_regex_lst,
        driver_name,
    ):
        """Init post post class.

        Args:
            post_file_name_identifier (str): Identifier for postprocessed files.
                                             The file prefix can contain regex expression and
                                            subdirectories.
            file_options_dict (dict): Dictionary with read-in options for
                                      the post_processed file. The respective child class will
                                       implement valid options for this dictionary.
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
            driver_name (str): Name of the associated driver.
        """
        self.files_to_be_deleted_regex_lst = files_to_be_deleted_regex_lst
        self.file_options_dict = file_options_dict
        self.driver_name = driver_name
        self.post_file_name_identifier = post_file_name_identifier
        self.post_file_path = None
        self.post_post_data = np.empty(shape=0)
        self.raw_file_data = None

    @classmethod
    def from_config_set_base_attributes(cls, config, driver_name):
        """Extract attributes of this base class from the config.

        Args:
            config (dict): Dictionary with problem description.
            driver_name (str): Name of driver that is used in this job-submission

        Returns:
            post_file_name_identifier (str): Identifier for postprocessed files.
                                             The file prefix can contain regex expression and
                                             subdirectories.
            file_options_dict (dict): Dictionary with read-in options for
                                      the post_processed file. The respective child class will
                                       implement valid options for this dictionary.
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
            driver_name (str): Name of the associated driver.
        """
        driver_params = config.get(driver_name)
        try:
            post_post_options = driver_params["driver_params"].get('post_post')
        except KeyError:
            post_post_options = driver_params.get("post_post")

        post_file_name_identifier = post_post_options.get('post_file_name_identifier')
        if not post_file_name_identifier:
            raise IOError(
                f"No option 'post_file_name_identifier' was provided in '{driver_name}' driver! "
                "PostPost object cannot be instantiated! Abort..."
            )
        if not isinstance(post_file_name_identifier, str):
            raise TypeError(
                "The option 'post_file_name_identifier' must be of type 'str' "
                f"but is of type {type(post_file_name_identifier)}. Abort..."
            )

        file_options_dict = post_post_options.get('file_options_dict')
        if not file_options_dict:
            raise IOError(
                f"No option 'file_options_dict' was provided in {driver_name} driver! "
                "PostPost object cannot be instantiated! Abort..."
            )
        if not isinstance(file_options_dict, dict):
            raise TypeError(
                "The option 'file_options_dict' must be of type 'dict' "
                f"but is of type {type(file_options_dict)}. Abort..."
            )

        files_to_be_deleted_regex_lst = post_post_options.get('files_to_be_deleted_regex_lst', [])
        if not isinstance(files_to_be_deleted_regex_lst, list):
            raise TypeError(
                "The option 'files_to_be_deleted_regex_lst' must be of type 'list' "
                f"but is of type {type(files_to_be_deleted_regex_lst)}. Abort..."
            )

        return post_file_name_identifier, file_options_dict, files_to_be_deleted_regex_lst

    def get_data_from_post_file(self, base_dir_post_file):
        """Get data of interest from postprocessed file.

        Args:
            base_dir_post_file (str): Path of the base directory that
                                           contains the file of interest.

        Returns:
            post_post_data (np.array): Final data from post post module
        """
        if not base_dir_post_file:
            raise ValueError(
                "The post_post processor requires a base_directory for the post "
                "processed files to operate on! Your input was empty! Abort..."
            )
        if not isinstance(base_dir_post_file, str):
            raise TypeError(
                "The argument 'base_dir_post_file' must be of type 'str' "
                f"but is of type {type(base_dir_post_file)}. Abort..."
            )

        post_file_path_regex = self._generate_path_to_post_file(base_dir_post_file)
        file_exists_bool = self._check_file_exist_and_is_unique(post_file_path_regex)
        breakpoint()
        if file_exists_bool:
            breakpoint()
            self._get_raw_data_from_file()
            self._filter_and_manipulate_raw_data()
            self._subsequent_data_manipulation()

        for file_path in self.files_to_be_deleted_regex_lst:
            self._clean_up(file_path)

        return self.post_post_data

    def _generate_path_to_post_file(self, base_dir_post_file):
        """Generate path to postprocessed file.

        Args:
            base_dir_post_file (str): Path to base directory that contains
                                      postprocessed file of interest

        Returns:
            post_file_path_regex (str): Path to postprocessed file that still
                                        contains wildcards or regex expressions
        """
        file_identifier = self.post_file_name_identifier
        post_file_path_regex = os.path.join(base_dir_post_file, file_identifier)
        return post_file_path_regex

    def _check_file_exist_and_is_unique(self, post_file_path_regex):
        """Check if file exists.

        Args:
            post_file_path_regex (str): Path to postprocessed file that still
                                        contains wildcards or regex expressions

        Returns:
            file_exists (bool): Boolean determine whether file exists. If true, the file
                                exists.
        """
        file_list = glob.glob(post_file_path_regex)
        if len(file_list) > 1:
            raise RuntimeError(
                "The post_post module found several files for the "
                "provided 'post_file_name_prefix'!"
                "The files are: {file_list}."
                "The file prefix must lead to a unique file. Abort..."
            )
        if len(file_list) == 1:
            self.post_file_path = file_list[0]
            file_exists = True
        else:
            _logger.warning(f"The file '{post_file_path_regex}' does not exist!")
            file_exists = False

        return file_exists

    @abc.abstractmethod
    def _get_raw_data_from_file(self):
        """Get the raw data from the files of interest."""

    @abc.abstractmethod
    def _filter_and_manipulate_raw_data(self):
        """Filter or clean the raw data for given criteria."""

    def _subsequent_data_manipulation(self):
        """Subsequent manipulate the post_post data.

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
