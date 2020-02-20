""" There should be a docstring """

import glob
from io import StringIO
import os
import numpy as np
import pandas as pd
from .post_post import Post_post


class PP_BACI_QoI(Post_post):
    """ Base class for post_post routines """

    def __init__(self, base_settings):

        super(PP_BACI_QoI, self).__init__(base_settings)
        self.num_post = base_settings['num_post']
        self.time_tol = base_settings['time_tol']
        self.target_time = base_settings['target_time']
        self.skiprows = base_settings['skiprows']

    @classmethod
    def from_config_create_post_post(cls, config, base_settings):
        """ Create post_post routine from problem description

        Args:
            config: input json file with problem description

        Returns:
            post_post: post_post object
        """
        post_post_options = base_settings['options']
        base_settings['num_post'] = len(config['driver']['driver_params']['post_process_options'])
        base_settings['target_time'] = post_post_options['target_time']
        base_settings['time_tol'] = post_post_options['time_tol']
        base_settings['skiprows'] = post_post_options['skiprows']
        return cls(base_settings)

    # ------------------------ COMPULSORY CHILDREN METHODS ------------------------
    def read_post_files(self):
        """ Loop over several post files of interest """

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

    def delete_field_data(self):
        """ Delete every output file except files with given prefix """

        inverse_prefix_expr = r"*[!" + self.file_prefix + r"]*"
        files_of_interest = os.path.join(self.output_dir, inverse_prefix_expr)
        post_file_list = glob.glob(files_of_interest)
        for filename in post_file_list:
            command_string = "rm " + filename
            # "cd " + self.output_file + "&& ls | grep -v --include=*.{mon,csv} | xargs rm"
            _, _, _ = self.run_subprocess(command_string)

    def error_handling(self):
        # TODO  ### Error Types ###
        # No QoI file
        # Time/Time step not reached
        # Unexpected values

        # Organized failed files
        input_file_extention = 'dat'
        if self.error is True:
            command_string = (
                "cd "
                + self.output_dir
                + "&& cd ../.. && mkdir -p postpost_error && cd "
                + self.output_dir
                + r"&& cd .. && mv *."
                + input_file_extention
                + r" ../postpost_error/"
            )
            _, _, _ = self.run_subprocess(command_string)
