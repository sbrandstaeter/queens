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

    def __init__(self, skip_rows, use_col, use_row, delete_data_flag, file_prefix):
        """ Init PostPost object

        Args:
            skip_rows (int):          Number of header rows to skip
            use_col (int):            Index of row to extract
            use_row (int):            Index of column to extract
            delete_data_flag (bool):  Delete files after processing
            file_prefix (str):        Prefix of result files

        """
        super(PostPostGeneric, self).__init__(delete_data_flag, file_prefix)

        self.skip_rows = skip_rows
        self.use_col = use_col
        self.use_row = use_row

    @classmethod
    def from_config_create_post_post(cls, options):
        """ Create post_post routine from problem description

        Args:
            options (dict): input options

        Returns:
            post_post: PostPostGeneric object
        """
        post_post_options = options['options']

        use_col = post_post_options['use_col']
        use_row = post_post_options['use_row']
        skip_rows = post_post_options['skiprows']
        delete_data_flag = post_post_options['delete_field_data']
        file_prefix = post_post_options['file_prefix']

        return cls(skip_rows, use_col, use_row, delete_data_flag, file_prefix)

    def read_post_files(self, files_of_interest):
        """ Loop over post files in given output directory """

        post_files_list = glob.glob(files_of_interest)
        # glob returns arbitrary list -> need to sort the list before using
        post_files_list.sort()
        post_out = np.empty(shape=0)

        for filename in post_files_list:
            try:
                post_data = pd.read_csv(
                    filename,
                    sep=r',|\s+',
                    usecols=[self.use_col],
                    skiprows=self.skip_rows,
                    engine='python',
                )
                quantity_of_interest = post_data.loc[self.use_row]
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
