"""Module for post-post processing of simulation results."""

import abc
import logging
import os

_logger = logging.getLogger(__name__)


class PostPost(metaclass=abc.ABCMeta):
    """Base class for post post processing.

    Attributes:
        post_post_files_regex_lst (lst): List with paths to postprocessed files.
                                            The file paths can contain regex expression.
        files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                The paths can contain regex expressions.
        driver_name (str): Name of the associated driver.
        raw_data_lst (lst): Pre-filtered list of raw-data objects from
                            postprocessed files.
    """

    def __init__(self, post_post_files_regex_lst, files_to_be_deleted_regex_lst, driver_name):
        """Init post post class.

        Args:
            post_post_files_regex_lst (lst): List with paths to postprocessed files.
                                             The file paths can contain regex expression.
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
            driver_name (str): Name of the associated driver.
        """
        self.post_post_files_regex_lst = post_post_files_regex_lst
        self.files_to_be_deleted_regex_lst = files_to_be_deleted_regex_lst
        self.driver_name = driver_name
        self.raw_data_lst = []

    @classmethod
    def from_config_create_post_post(config, driver_name):
        """Create PostPost object from problem description.

        Args:
            config (dict): input json file with problem description
            driver_name (str): Name of driver that is used in this job-submission

        Returns:
            post_post (obj): post_post object
        """
        from .post_post_baci import PostPostBACI

        post_post_dict = {
            'baci': PostPostBACI,
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

        post_post_version = post_post_options.get('post_post_approach_sel')
        if not post_post_version:
            raise ValueError(
                "The posst_post section did not specify a valid 'post_post_approach_sel'! "
                f"Valid options are {post_post_dict.keys()}. Abort..."
            )

        post_post_class = post_post_dict[post_post_version]
        post_post = post_post_class.from_config_create_post_post(config, driver_name)

        return post_post

    def get_data_from_post_files(self):
        """Get data of interest from postprocessed files.

        Args:
            file_paths_lst (list): List with paths to post-processed files.

        Returns:
            None
        """
        for file_path in self.post_post_files_regex_lst:
            PostPost._check_file_exist(file_path)
            self._get_raw_data_from_file(file_path)
            self._check_raw_data()
            self._manipulate_raw_data()

        for file_path in self.files_to_be_deleted_regex_lst:
            PostPost._clean_up(file_path)

    @staticmethod
    def _check_files_exist(file_path):
        """Check if file exists.

        Args:
            file_path (str): Path to file of interest. The path might contain
                             regex expressions.

        Returns:
            None
        """
        file_exists = os.path.isfile(file_path)
        if not file_exists:
            raise FileNotFoundError(f"The file '{file_path}' does not exist! Abort...")

    @abc.abstractmethod
    def _get_raw_data_from_file(self):
        pass

    @abc.abstractmethod
    def _check_raw_data(self):
        pass

    @abc.abstractmethod
    def _manipulate_raw_data(self):
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
