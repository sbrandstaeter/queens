import glob
from io import StringIO
import os
import numpy as np
import pandas as pd
import pyansys
from pqueens.post_post.post_post import PostPost


class PostPostANSYS(PostPost):
    """ Class for processing ANSYS file

        Attributes:
            skiprows (int):  Number of rows to skip in result file


    """

    def __init__(self, usecols, delete_data_flag, file_prefix):
        """ Init PostPost object

            Args:
                usecols (list):           Index of columns to use in results
                delete_data_flag (bool):  Delete files after processing
                file_prefix (str):        Prefix of result files

        """
        self.usecols = usecols
        super(PostPostANSYS, self).__init__(delete_data_flag, file_prefix)

    @classmethod
    def from_config_create_post_post(cls, options):
        """ Create post_post routine from problem description

        Args:
            options (dict): input options

        Returns:
            post_post: PostPostANSYS object
        """
        post_post_options = options['options']
        usecols = post_post_options['usecols']
        delete_data_flag = post_post_options['delete_field_data']
        file_prefix = post_post_options['file_prefix']

        return cls(usecols, delete_data_flag, file_prefix)

    def read_post_files(self, file_names, **kwargs):
        """
        Loop over post files in given output directory

        Args:
            file_names (str): Path with filenames without specific extension

        Returns:
            None

        """

        post_files_list = glob.glob(file_names)
        # glob returns arbitrary list -> need to sort the list before using
        post_files_list.sort()
        post_out = np.empty(shape=0)

        for filename in post_files_list:
            try:
                post_data = pyansys.read_binary(filename)
                nnum, qoi_array = post_data.nodal_solution(0)
                quantity_of_interest = qoi_array[self.usecols[0], self.usecols[1]]
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
