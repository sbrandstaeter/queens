import glob
from io import StringIO
import os
import numpy as np
import pandas as pd
from pqueens.post_post.post_post import PostPost


class PostPostBACI(PostPost):
    """ Class for post-post-processing BACI output

        Attributes:
            num_post (int):         TODO what is this? Where is this needed?
            time_tol (float):       Tolerance if desired time can not be matched exactly
            target_time (float):    Time at which to evaluate QoI
            skiprows (int):         Number of header rows to skip

    """

    def __init__(self, num_post, time_tol, target_time, skiprows,
                 usecols, delete_data_flag, file_prefix):
        """ Init PostPost object

        Args:
            num_post (int):           TODO what is this? Where is this needed?
            time_tol (float):         Tolerance if desired time can not be matched exactly
            target_time (float):      Time at which to evaluate QoI
            skiprows (int):           Number of header rows to skip
            usecols (list):           Index of columns to use in result file
            delete_data_flag (bool):  Delete files after processing
            file_prefix (str):        Prefix of result files

        """

        super(PostPostBACI, self).__init__(usecols, delete_data_flag, file_prefix)

        self.num_post = num_post
        self.time_tol = time_tol
        self.target_time = target_time
        self.skiprows = skiprows

    @classmethod
    def from_config_create_post_post(cls, config, base_settings):
        """ Create post_post routine from problem description

        Args:
            config (dict): input json file with problem description
            base_settings (dict): TODO what is this?? why are there two dicts?

        Returns:
            post_post: PostPostBACI object
        """
        post_post_options = base_settings['options']
        # TODO what is this? Where is this needed?
        num_post = len(config['driver']['driver_params']['post_process_options'])
        time_tol = post_post_options['time_tol']
        target_time = post_post_options['target_time']
        skiprows = post_post_options['skiprows']
        usecols = post_post_options['usecols']
        delete_data_flag = post_post_options['delete_field_data']
        file_prefix = post_post_options['file_prefix']

        return cls(num_post, time_tol, target_time, skiprows,
                   usecols, delete_data_flag, file_prefix)

    def read_post_files(self):
        """ Loop over post files in given output directory """

        prefix_expr = '*' + self.file_prefix + '*'
        files_of_interest = os.path.join(self.output_dir, prefix_expr)
        post_files_list = glob.glob(files_of_interest)
        post_out = []

        for filename in post_files_list:
            try:
                post_data = pd.read_csv(
                    filename,
                    sep=r',|\s+',
                    usecols=self.usecols,
                    skiprows=self.skiprows,
                    engine='python',
                )
                identifier = abs(post_data.iloc[:, 0] - self.target_time) < self.time_tol
                quantity_of_interest = post_data.loc[identifier].iloc[0, 1]
                post_out = np.append(post_out, quantity_of_interest)
                # very simple error check
                if not post_out:
                    self.error = True
                    self.result = None
                    break
            except IOError:
                self.error = True
                self.result = None
                break
        self.error = False
        self.result = post_out
