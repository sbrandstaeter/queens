"""Module for post-post processing of simulation results."""

import abc
import logging
import os

_logger = logging.getLogger(__name__)


class PostPost(metaclass=abc.ABCMeta):
    """Base class for post post processing.

    Attributes:
        post_post_file_path (str): Path to postprocessed files.
                                    The file path can contain regex expression.
        file_options (dict): Dictionary with read-in options for
                                the post_processed file
        files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                The paths can contain regex expressions.
        driver_name (str): Name of the associated driver.
        post_post_data (np.array): Cleaned, filtered or manipulated post_post data
    """

    def __init__(
        self,
        post_post_file_path,
        file_options_dict,
        files_to_be_deleted_regex_lst,
        driver_name,
    ):
        """Init post post class.

        Args:
            post_post_file_path (str): Path to postprocessed files.
                                       The file path can contain regex expression.
            file_options_dict (dict): Dictionary with read-in options for
                                      the post_processed file
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
            driver_name (str): Name of the associated driver.
        """
        self.post_post_file_path = post_post_file_path
        self.files_to_be_deleted_regex_lst = files_to_be_deleted_regex_lst
        self.file_options_dict = file_options_dict
        self.driver_name = driver_name
        self.post_post_data = None
        self.raw_file_data = None

    @classmethod
    def from_config_create_post_post(config, driver_name):
        """Create PostPost object from problem description.

        Args:
            config (dict): input json file with problem description
            driver_name (str): Name of driver that is used in this job-submission

        Returns:
            post_post (obj): post_post object
        """
        from .post_post_csv_data import PostPostCsv

        post_post_dict = {
            'csv': PostPostCsv,
        }

        driver_params = config.get(driver_name)
        if not driver_params:
            raise ValueError(
                "No driver parameters found in problem description! "
                f"You specified the driver name '{driver_name}', "
                "which could not be found in the problem description. Abort..."
            )

        post_post_options = driver_params.get('post_post')
        if not post_post_options:
            raise ValueError(
                "The 'post_post' options were not found in the driver '{driver_name}'! Abort..."
            )

        post_post_file_path = post_post_options.get('post_post_file_path')
        if not post_post_file_path:
            raise IOError(
                f"No option 'post_post_files_regex_lst' was provided in {driver_name} driver! "
                "PostPost object cannot be instantiated! Abort..."
            )
        if not isinstance(post_post_file_path, str):
            raise TypeError(
                "The option 'post_post_file_path' must be of type 'str' "
                f"but is of type {type(post_post_file_path)}. Abort..."
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

        post_post_version = post_post_options.get('post_post_approach_sel')
        if not post_post_version:
            raise ValueError(
                "The posst_post section did not specify a valid 'post_post_approach_sel'! "
                f"Valid options are {post_post_dict.keys()}. Abort..."
            )

        post_post_class = post_post_dict[post_post_version]
        post_post = post_post_class.from_config_create_post_post(
            driver_name,
            post_post_file_path,
            file_options_dict,
            files_to_be_deleted_regex_lst,
        )

        return post_post

    def get_data_from_post_file(self):
        """Get data of interest from postprocessed file.

        Returns:
            None
        """
        self._check_file_exist()
        self._get_raw_data_from_file()
        self._filter_raw_data()
        self._manipulate_data()

        for file_path in self.files_to_be_deleted_regex_lst:
            PostPost._clean_up(file_path)

    def _check_files_exist(self):
        """Check if file exists.

        Returns:
            None
        """
        file_exists = os.path.isfile(self.post_post_files_path)
        if not file_exists:
            raise FileNotFoundError(
                f"The file '{self.post_post_files_path}' does not exist! Abort..."
            )

    @abc.abstractmethod
    def _get_raw_data_from_file(self):
        """Get the raw data from the files of interest."""
        pass

    @abc.abstractmethod
    def _filter_raw_data(self):
        """Filter or clean the raw data for given criteria."""
        pass

    @abc.abstractmethod
    def _manipulate_data(self):
        """Manipulate the post_post data."""
        pass

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
        except (OSError, FileNotFoundError):
            _logger.debug(f"Could not remove file with path: '{file_path}'. Skip...")
