""" There should be some docstring """

import os.path
import pandas as pd
from pqueens.post_post.post_post import Post_post


class PP_time_series(Post_post):
    """ Base class for post_post routines """

    def __init__(self, base_settings):
        super(PP_time_series, self).__init__(base_settings)
        self.file = base_settings['file_name']

    @classmethod
    def from_config_create_post_post(cls, config, base_settings):
        """ Create post_post routine from problem description

        Args:
            config: input json file with problem description

        Returns:
            post_post: post_post object
        """

        post_post_options = config['driver']['driver_params']['post_post']
        base_settings['file_name'] = post_post_options['file_name']
        return cls(base_settings)

    def read_post_files(self, output_file):  # output file given by driver
        # loop over several post files if list of post processors given
        output_dir = os.path.dirname(output_file).strip('vtu')
        path = output_dir + self.file
        post_data = pd.read_csv(path, usecols=self.usecols, sep='\s+', skiprows=self.skiprows)
        post_out = post_data[(post_data.iloc[:, 0] >= 4) & (post_data.iloc[:, 0] <= 7)].to_numpy()
        post_out = post_out.copy(order='C')
        if not post_out.any():  # timestep reached? <=> variable is empty?
            self.error = True
        return post_out, self.error
