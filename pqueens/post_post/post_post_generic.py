import glob
from io import StringIO
import os
import numpy as np
import pandas as pd
from pqueens.post_post.post_post import PostPost


class PostPostGeneric(PostPost):
    """ Class for post-post-processing output from a csv file

        Attributes:
            use_col (float):    Index of row to extract
            use_row (float):    Index of column to extract

    """

    def __init__(self, use_col, use_row, delete_data_flag, file_prefix):
        """ Init PostPost object

        Args:
            use_col (int):            Index of row to extract
            use_row (int):            Index of column to extract
            delete_data_flag (bool):  Delete files after processing
            file_prefix (str):        Prefix of result files

        """
        # TODO pass empty list for now 
        super(PostPostGeneric, self).__init__([], delete_data_flag, file_prefix)

        self.use_col = use_col
        self.use_row = use_row


    @classmethod
    def from_config_create_post_post(cls, config, base_settings):
        """ Create post_post routine from problem description

        Args:
            config (dict): input json file with problem description
            base_settings (dict): TODO what is this?? why are there two dicts?

        Returns:
            post_post: PostPostGeneric object
        """
        post_post_options = base_settings['options']

        use_col = post_post_options['use_col']
        use_row = post_post_options['use_row']
        delete_data_flag = post_post_options['delete_field_data']
        file_prefix = post_post_options['file_prefix']

        return cls(use_col, use_row, delete_data_flag, file_prefix)

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
                    usecols=self.use_col,
                    skiprows=0,
                    engine='python',
                )
                identifier = abs(post_data.iloc[:, 0] - self.target_time) < self.time_tol
                quantity_of_interest = post_data.loc[self.use_col]
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
