import abc
import numpy as np
import pandas as pd
import os.path

class Post_post(metaclass=abc.ABCMeta):
    """ Base class for post_post routines """

    def __init__(self, base_settings):

        self.target_time = base_settings['target_time']
        self.time_tol = base_settings['time_tol']
        self.subfix = base_settings['subfix']
        self.usecols = base_settings['usecols']
        self.skiprows = base_settings['skiprows']
        self.error = False
        self.num_post = base_settings['num_post']
        self.delete_field_data = base_settings['delete_field_data']

    @classmethod
    def from_config_create_post_post(cls, config):
        """ Create post_post routine from problem description

        Args:
            config: input json file with problem description

        Returns:
            post_post: post_post object
        """

        # determine which object to create
        post_post_options = config['driver']['driver_params']['post_post']

#### create base settings ##################
        base_settings={}
        base_settings['target_time'] = post_post_options['target_time']
        base_settings['time_tol'] = post_post_options['time_tol']
        base_settings['subfix'] = post_post_options['subfix']
        base_settings['usecols'] = post_post_options['usecols']
        base_settings['skiprows'] = post_post_options['skiprows']
        base_settings['delete_field_data'] = post_post_options['delete_field_data']
        base_settings['num_post'] = len(config['driver']['driver_params']['post_process_options'])
#### end base settings #####################
        return cls(base_settings)

    def read_post_files(self, output_file): # output file given by driver
    # loop over several post files if list of post processors given
        output_dir = os.path.dirname(output_file)
        post_out = []

        for num in range(self.num_post):
            # different read methods depending on subfix
            if self.subfix=='mon':
                path = output_dir + r'/QoI_' + str(num+1) + r'.mon'
                post_data = np.loadtxt(path, usecols=self.usecols, skiprows=self.skiprows)
            elif self.subfix=='csv':
                path =output_dir + r'/QoI_' + str(num+1) + r'.csv'
                post_data = pd.read_csv(path, usecols=self.usecols, skiprows=self.skiprows)
            else:
                raise RuntimeError("Subfix of post processed file is unknown!")

            QoI_identifier = abs(post_data[:,0]-self.target_time) < self.time_tol
            QoI = post_data[QoI_identifier][0,1]
            post_out = np.append(post_out, QoI) # select only row with timestep equal to target time step
            if not post_out: # timestep reached? <=> variable is empty?
                self.error = True

        return post_out, self.error
