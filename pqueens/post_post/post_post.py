import abc
import numpy as np
import pandas as pd
import os.path

class Post_post(metaclass=abc.ABCMeta):
    """ Base class for post_post routines """

    def __init__(self, base_settings):

        self.target_step = base_settings['target_step']
        self.subfix = base_settings['subfix']
        self.usecols = base_settings['usecols']
        self.skiprows = base_settings['skiprows']
        self.error = False
        self.num_post = base_settings['num_post']

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
        base_settings['target_step'] = post_post_options['target_step']
        base_settings['subfix'] = post_post_options['subfix']
        base_settings['usecols'] = post_post_options['usecols']
        base_settings['skiprows'] = post_post_options['skiprows']
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
                post_data = np.loadtxt(output_dir + r'/QoI_' + num + r'.mon', usecols=self.usecols, skiprows=self.skiprows)
            elif self.subfix=='csv':
                post_data = pd.read_csv(output_dir + r'/QoI_' + num + r'.csv', usecols=self.usecols, skiprows=self.skiprows)
            else:
                raise RuntimeError("Subfix of post processed file is unknown!")

            post_out = np.append[post_out, post_data[post_data[:,0]==self.target_step]] # select only row with timestep equal to target time step
            if not post_out: # timestep reached? <=> variable is empty?
                self.error = True

        return post_out, self.error
