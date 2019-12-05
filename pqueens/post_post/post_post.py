""" This should be a docstring """

import abc
import subprocess
import os
import glob


class PostPost(metaclass=abc.ABCMeta):
    """ Base class for post post processing

        Attributes:
            usecols ():
            delete_data_flag ():
            file_prefix ():
            error ():
            output_dir ():
            result ():

    """

    def __init__(self, usecols, delete_data_flag, file_prefix):
        """ Init post post class

            Args:
                usecols (list):
                delete_data_flag ():
                file_prefix ():

        """

        self.usecols = usecols
        self.delete_data_flag = delete_data_flag
        self.file_prefix = file_prefix

        self.error = False
        self.output_dir = None
        self.result = None

    @classmethod
    def from_config_create_post_post(cls, config):
        """ Create PostPost object from problem description

        Args:
            config (dict): input json file with problem description

        Returns:
            post_post: post_post object
        """

        from .post_post_ansys import PostPostANSYS
        from .post_post_baci import PostPostBACI
        from .post_post_deal import PostPostDEAL

        post_post_dict = {
            'ansys': PostPostANSYS,
            'baci': PostPostBACI,
            'deal': PostPostDEAL,
        }

        # determine which object to create
        # TODO this is not a reliable approach? What if we have multiple drivers?
        # However, this cannot be fixed by itself here, but we need to 
        # cleanup the whole input parameter handling to fix this.
        post_post_options = config['driver']['driver_params']['post_post']

        if post_post_options['post_post_approach_sel'] == 'ansys':
            post_post_version = 'ansys'
        elif post_post_options['post_post_approach_sel'] == 'baci':
            post_post_version = 'baci'
        elif post_post_options['post_post_approach_sel'] == 'deal':
            post_post_version = 'deal'
        else:
            raise RuntimeError("post_post_approach_sel not set, fix your input file")

        post_post_class = post_post_dict[post_post_version]

        # ---------------------------- CREATE BASE SETTINGS ---------------------------
        base_settings = {}
        base_settings['options'] = post_post_options
        base_settings['file_prefix'] = post_post_options['file_prefix']
        base_settings['usecols'] = post_post_options['usecols']
        base_settings['delete_field_data'] = post_post_options['delete_field_data']

        # TODO remove "base_settings" and reintroduce name"
        post_post = post_post_class.from_config_create_post_post(config, base_settings)
        return post_post

    def error_handling(self):
        """ TODO Complete docstring 

            What does this function do
        """
        # TODO  ### Error Types ###
        # No QoI file
        # Time/Time step not reached
        # Unexpected values

        # Organized failed files
        input_file_extention = 'dat'
        # TODO add documentation what happens here?
        # Make this platform independent 
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

    def delete_field_data(self):
        """ Delete every output file except files with given prefix """

        inverse_prefix_expr = r"*[!" + self.file_prefix + r"]*"
        files_of_interest = os.path.join(self.output_dir, inverse_prefix_expr)
        post_file_list = glob.glob(files_of_interest)
        for filename in post_file_list:
            os.remove(filename)

    def postpost_main(self, output_dir):
        """ This should be a docstring """
        # TODO add meaningful docsctring 
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

    # ----------------------------- AUXILARY FUNCTION -----------------------------
    def run_subprocess(self, command_string):
        """ Method to run command_string outside of Python """
        process = subprocess.Popen(
            command_string,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True,
        )
        stdout, stderr = process.communicate()
        process.poll()
        return stdout, stderr, process
