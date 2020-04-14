""" This should be a docstring """

import abc
import subprocess
import os
import glob


class PostPost(metaclass=abc.ABCMeta):
    """ Base class for post post processing

        Attributes:
            delete_data_flag (bool):  Delete files after processing
            file_prefix ():           Prefix of result files
            error ():
            output_dir (str):         Path to result files
            result ():

    """

    def __init__(self, delete_data_flag, file_prefix):
        """ Init post post class

            Args:
                delete_data_flag (bool): Delete files after processing
                file_prefix (str):       Prefix of result files

        """

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
        from .post_post_generic import PostPostGeneric
        from .post_post_baci_shape import PostPostBACIShape

        post_post_dict = {
            'ansys': PostPostANSYS,
            'baci': PostPostBACI,
            'deal': PostPostDEAL,
            'generic': PostPostGeneric,
            'baci_shape': PostPostBACIShape
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
        elif post_post_options['post_post_approach_sel'] == 'generic':
            post_post_version = 'generic'
        elif post_post_options['post_post_approach_sel'] == 'baci_shape':
            post_post_version = 'baci_shape'
        else:
            raise RuntimeError("post_post_approach_sel not set, fix your input file")

        post_post_class = post_post_dict[post_post_version]

        # ---------------------------- CREATE BASE SETTINGS ---------------------------
        base_settings = {}
        base_settings['options'] = post_post_options
        #base_settings['file_prefix'] = post_post_options['file_prefix']
        #base_settings['usecols'] = post_post_options['usecols']
        #base_settings['delete_field_data'] = post_post_options['delete_field_data']

        post_post = post_post_class.from_config_create_post_post(base_settings)
        return post_post

    def error_handling(self):
        """ Mark failed simulation and set results appropriately

            What does this function do?? This is super unclear
        """
        # TODO  ### Error Types ###
        # No QoI file
        # Time/Time step not reached
        # Unexpected values

        # Organized failed files
        input_file_extention = 'dat'
        # TODO add documentation what happens here?
        # TODO Make this platform independent
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
        """ Delete output files except files with given prefix """

        inverse_prefix_expr = r"*[!" + self.file_prefix + r"]*"
        files_of_interest = os.path.join(self.output_dir, inverse_prefix_expr)
        post_file_list = glob.glob(files_of_interest)
        for filename in post_file_list:
            os.remove(filename)

    def postpost_main(self, output_dir):
        """ Method that coordinates post post processing

            Args:
                output_dir (str): Path to output directory

            Returns:
                result of post_post
                # TODO determine type
        """

        self.output_dir = output_dir
        self.read_post_files()
        # mark failed simulation and set results appropriately
        self.error_handling()
        # TODO check if json input is interpreated as boolean
        if self.delete_data_flag:
            self.delete_field_data()

        return self.result

    @abc.abstractmethod
    def read_post_files(self):
        """ This method has to be implemented by all child classes """
        pass

    def run_subprocess(self, command_string):
        """ Method to run command_string outside of Python

            Args:
                command_string (str): Command to be executed

            Returns:
                str, str: stdout and std error
        """
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
