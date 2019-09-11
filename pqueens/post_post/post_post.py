""" This should be a docstring """

import abc
import subprocess


class Post_post(metaclass=abc.ABCMeta):
    """ Base class for post_post routines """

    def __init__(self, base_settings):
        self.usecols = base_settings['usecols']
        self.error = False
        self.delete_data_flag = base_settings['delete_field_data']
        self.file_prefix = base_settings['file_prefix']
        self.output_dir = None
        self.result = None

    @classmethod
    def from_config_create_post_post(cls, config):
        """ Create post_post routine from problem description

        Args:
            config: input json file with problem description

        Returns:
            post_post: post_post object
        """

        from .pp_BACI_QoI import PP_BACI_QoI
        from .pp_time_series import PP_time_series

        post_post_dict = {'baci_qoi': PP_BACI_QoI,
                          'time_series': PP_time_series}

        # determine which object to create
        post_post_options = config['driver']['driver_params']['post_post']
        if 'type' in post_post_options:
            post_post_version = post_post_options['type']
        else:
            post_post_version = 'baci_qoi'  # set baci analysis as default
        post_post_class = post_post_dict[post_post_version]

# ---------------------------- CREATE BASE SETTINGS ---------------------------
        base_settings = {}
        base_settings['options'] = post_post_options
        base_settings['file_prefix'] = post_post_options['file_prefix']
        base_settings['usecols'] = post_post_options['usecols']
        base_settings['delete_field_data'] = post_post_options['delete_field_data']

# ----------------------------- END BASE SETTINGS -----------------------------
        post_post = post_post_class.from_config_create_post_post(config, base_settings)
        return post_post

    def postpost_main(self, output_dir):
        """ This should be a docstring """
        self.output_dir = output_dir
        self.read_post_files()
        self.error_handling()  # mark failed simulation and set results approp.
        if self.delete_data_flag:  # TODO check if json input is interpreated as boolean
            self.delete_field_data()
        return self.result

# ------------------------ COMPULSORY CHILDREN METHODS ------------------------
    @abc.abstractmethod
    def read_post_files(self):
        """ This should be a docstring """
        pass

    @abc.abstractmethod
    def delete_field_data(self):
        """ This should be a docstring """
        pass

    @abc.abstractmethod
    def error_handling(self):
        """ This should be a docstring """
        pass
# ----------------------------- AUXILARY FUNCTION -----------------------------
    def run_subprocess(self, command_string):
        """ Method to run command_string outside of Python """
        process = subprocess.Popen(command_string,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True,
                                   universal_newlines=True)
        stdout, stderr = process.communicate()
        process.poll()
        return stdout, stderr, process
