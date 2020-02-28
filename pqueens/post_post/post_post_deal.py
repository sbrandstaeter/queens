""" There should be some docstring """

import glob
import os.path
import pandas as pd
import numpy as np
from pqueens.post_post.post_post import PostPost


class PostPostDEAL(PostPost):
    """ Base class for post_post routines """

    def __init__(self, skiprows, usecols, delete_data_flag, file_prefix):
        """ Init Object
        
            Args:
                skiprows (int):              Number of header rows to skip
                usecols (int):               Index of columns to use in result file
                delete_data_flag (bool):     Delete files after processing
                file_prefix (str):           Prefix of result file

        """
        super(PostPostDEAL, self).__init__(delete_data_flag, file_prefix)

        self.usecols = usecols
        self.skiprows = skiprows

    @classmethod
    def from_config_create_post_post(cls, options):
        """ Create post_post routine from problem description

        Args:
            options (dict): input options

        Returns:
            post_post: post_post object
        """
        post_post_options = options['options']
        skiprows = post_post_options['skiprows']
        usecols = post_post_options['usecols']
        delete_data_flag = post_post_options['delete_field_data']
        file_prefix = post_post_options['file_prefix']

        return cls(skiprows, usecols, delete_data_flag, file_prefix)

    def read_post_files(self):
        """ Loop over post files in given output directory """

        prefix_expr = '*' + self.file_prefix + '*'
        files_of_interest = os.path.join(self.output_dir, prefix_expr)
        post_files_list = glob.glob(files_of_interest)
        # TODO this is not general but only for navier stokes solver
        path = post_files_list[0]
        post_out = []

        try:
            post_data = pd.read_csv(path, usecols=self.usecols,
                                    sep=r'\s+', skiprows=self.skiprows)
            post_out = post_data[
                (post_data.iloc[:, 0] >= 4) & (post_data.iloc[:, 0] <= 7)
            ].to_numpy()
            post_out = post_out.copy(order='C')
            # ------------- TODO THIS IS JUST A WORKAROUND TO EXTRACT ONE QOI -------------
            self.result = np.max(post_out[:, 1])
            # ------------------------------- END WORKAROUND ------------------------------
            if not post_out.any():  # timestep reached? <=> variable is empty?
                self.error = True
                self.result = None
        except RuntimeError:
            self.error = True
            self.result = None

        if not post_out.any():  # timestep reached? <=> variable is empty?
            self.error = True
            self.result = None
