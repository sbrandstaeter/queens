""" There should be a docstring """

import glob
from io import StringIO
import os
import numpy as np
import pandas as pd
import pyansys
from pqueens.post_post.post_post import PostPost


class PostPostANSYS(PostPost):
    """ Base class for post_post routines """

    def __init__(self, skiprows, usecols, delete_data_flag, file_prefix):

        super(PostPostANSYS, self).__init__(usecols, delete_data_flag, file_prefix)

        self.skiprows = skiprows

    @classmethod
    def from_config_create_post_post(cls, config, base_settings):
        """ Create post_post routine from problem description

        Args:
            config: input json file with problem description

        Returns:
            post_post: PostPostANSYS object
        """
        post_post_options = base_settings['options']
        skiprows = post_post_options['skiprows']
        usecols = post_post_options['usecols']
        delete_data_flag = post_post_options['delete_field_data']
        file_prefix = post_post_options['file_prefix']

        return cls(skiprows, usecols, delete_data_flag, file_prefix)

    # ------------------------ COMPULSORY CHILDREN METHODS ------------------------
    def read_post_files(self):
        """ Loop over several post files of interest """

        prefix_expr = '*' + self.file_prefix + '*'
        files_of_interest = os.path.join(self.output_dir, prefix_expr)
        post_files_list = glob.glob(files_of_interest)
        post_out = []

        for filename in post_files_list:
            try:
                post_data = pyansys.read_binary(filename)
                nnum, qoi_array = post_data.nodal_solution(self.skiprows)
                quantity_of_interest = qoi_array[self.usecols[0], self.usecols[1]]
                post_out = np.append(post_out, quantity_of_interest)
                # select only row with timestep equal to target time step
                if not post_out:  # timestep reached? <=> variable is empty?
                    self.error = True
                    self.result = None
                    break
            except IOError:
                self.error = True  # TODO in the future specify which error type
                self.result = None
                break
        self.error = False
        self.result = post_out
