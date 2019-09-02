import abc
import numpy as np
import pandas as pd
import os.path

class Post_post(metaclass=abc.ABCMeta):
    """ Base class for post_post routines """
    #TODO: #### FOR  CLUSTER JOBS WE NEED TO COPY THE ENTIRE POST_POST DIR #####

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
        from pqueens.post_post.pp_BACI_QoI import PP_BACI_QoI
        from pqueens.post_post.pp_time_series import PP_time_series

        post_post_dict = {'baci_qoi': PP_BACI_QoI,
                          'time_series': PP_time_series}

        # determine which object to create
        post_post_options = config['driver']['driver_params']['post_post']
        if 'type' in post_post_options:
            post_post_version = post_post_options['type']
        else: post_post_version = 'baci_qoi' # set baci analysis as default
        post_post_class = post_post_dict[post_post_version]

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
        post_post = post_post_class.from_config_create_post_post(config, base_settings)
        return post_post

    @abc.abstractmethod
    def read_post_files(self, output_file=None): # output file might be given by driver
        pass

