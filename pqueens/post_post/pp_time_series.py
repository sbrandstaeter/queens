""" There should be some docstring """

import glob
import os.path
import pandas as pd
import numpy as np
from pqueens.post_post.post_post import Post_post


class PP_time_series(Post_post):
    """ Base class for post_post routines """

    def __init__(self, base_settings):
        super(PP_time_series, self).__init__(base_settings)
#        self.num_post = base_settings['num_post']
#        self.time_tol = base_settings['time_tol']
#        self.target_time = base_settings['target_time']
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
#        base_settings['num_post'] = len(config['driver']['driver_params']['post_process_options'])
        base_settings['target_time'] = post_post_options['target_time']
#        base_settings['time_tol'] = post_post_options['time_tol']
        base_settings['skiprows'] = post_post_options['skiprows']
        return cls(base_settings)

    def read_post_files(self):  # output file given by driver

        prefix_expr = self.file_prefix + '*'
        files_of_interest = os.path.join(self.output_dir, prefix_expr)
        post_files_list = glob.glob(files_of_interest)
        path = post_files_list[0]
        post_out = []

        try:
            post_data = pd.read_csv(path, usecols=self.usecols, sep='\s+', skiprows=self.skiprows)
            post_out = post_data[(post_data.iloc[:, 0] >= 4) & (post_data.iloc[:, 0] <= 7)].to_numpy()
            post_out = post_out.copy(order='C')
    # ------------- TODO THIS IS JUST A WORKAROUND TO EXTRACT ONE QOI -------------
            self.result = np.max(post_out[:, 1])
    # ------------------------------- END WORKAROUND ------------------------------
            if not post_out.any():  # timestep reached? <=> variable is empty?
                self.error = True
        except RuntimeError:
            self.error = True
            self.result = None

        if not post_out.any():  # timestep reached? <=> variable is empty?
            self.error = True
            self.result = None

    def delete_field_data(self):
        """ Delete every output file except files with given prefix """

        inverse_prefix_expr = r"[!" + self.file_prefix + r"]*"
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
        pass
