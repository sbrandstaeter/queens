import glob
import numpy as np
import pandas as pd
from pqueens.post_post.post_post import PostPost
import logging
_logger = logging.getLogger(__name__)


class PostPostBACIVector(PostPost):
    """ Class for post-post-processing vector-valued BACI output

        Attributes:
            time_tol (float):       Tolerance if desired time can not be matched exactly
            target_time (float):    Time at which to evaluate QoI
            skiprows (int):         Number of header rows to skip

    """

    def __init__(self, time_tol, target_time, skiprows, usecols, delete_data_flag, file_prefix):
        """ Init PostPost object

        Args:
            time_tol (float):         Tolerance if desired time can not be matched exactly
            target_time (float):      Time at which to evaluate QoI
            skiprows (int):           Number of header rows to skip
            usecols (list):           Index of columns to use in result file
            delete_data_flag (bool):  Delete files after processing
            file_prefix (str):        Prefix of result files

        """

        super(PostPostBACIVector, self).__init__(delete_data_flag, file_prefix)
        self.usecols = usecols
        self.time_tol = time_tol
        self.target_time = target_time
        self.skiprows = skiprows

    @classmethod
    def from_config_create_post_post(cls, options):
        """ Create post_post routine from problem description

        Args:
            options (dict): input options

        Returns:
            post_post: PostPostBACI object
        """
        post_post_options = options['options']
        time_tol = post_post_options['time_tol']
        target_time = post_post_options['target_time']
        skiprows = post_post_options['skiprows']
        usecols = post_post_options['usecols']
        delete_data_flag = post_post_options['delete_field_data']
        file_prefix = post_post_options['file_prefix']

        return cls(time_tol, target_time, skiprows, usecols, delete_data_flag, file_prefix)

    def read_post_files(self, file_names, **kwargs):
        """
        Loop over post files in given output directory

        Args:
            file_names (str): Path with filenames without specific extension

        Returns:
            None

        """
        post_files_list = glob.glob(file_names)
        post_out = np.empty(shape=0)

        for filename in post_files_list:
            try:
                post_data = pd.read_csv(
                    filename,
                    sep=r',|\s+',
                    usecols=self.usecols,
                    skiprows=self.skiprows,
                    engine='python',
                )
            except IOError:
                _logger.info("Could not read csv-file.")
                self.error = True
                self.result = None
                break

            quantity_of_interest = post_data
            post_out = np.append(post_out, quantity_of_interest)

            # very simple error check
            if not np.any(post_out):
                self.error = True
                self.result = None
                break

        self.error = False
        self.result = post_out
