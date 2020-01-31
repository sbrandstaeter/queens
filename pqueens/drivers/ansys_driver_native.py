""" This should be a docstring """

import os
from pqueens.drivers.driver import Driver


class AnsysDriverNative(Driver):
    """ Driver to run ANSYS natively on workstation

        Attributes:
            mpi_config (dict): TODO unclear what this is needed for 

    """

    def __init__(self, base_settings):
        # TODO dunder init should not be called with dict 
        super(AnsysDriverNative, self).__init__(base_settings)
        #self.mpi_config = {}

    @classmethod
    def from_config_create_driver(cls, config, base_settings, workdir=None):
        """ Create Driver from input file

        Args:
            config (dict):          Input options
            base_settings (dict):   Second dict with input options TODO should probably be removed
            workdir (str):          Probably not used TODO remove 

        Returns:
            driver: AnsysDriverNative object
        """
        base_settings['address'] = 'localhost:27017'
        return cls(base_settings)

    def setup_dirs_and_files(self):
        """ Setup directory structure

            Args:
                driver_options (dict): Options dictionary

            Returns:
                str, str, str: simualtion prefix, name of input file, name of output file
        """
        # TODO this really necesarry? Why dont we have a seperate input parameter for this?
        # split ANSYS excecutable into main and custom executable (if present)
        # Note that, in the JSON file section "path_to_executable", the main and
        # customized executable have to be input as follows:
        # "\"main_executable\"-\"custom_executable\"" (i.e., both excecutables
        # within quotation marks and separated by a hyphen)
        # If the standard ANSYS executable is supposed to be used, the hyphen
        # must not be forgot at the end of the input, though:
        # "\"main_executable\"-".
        # self.main_executable, self.custom_executable = self.executable.split('-')
        self.main_executable = self.executable

        # base directories
        dest_dir = os.path.join(str(self.experiment_dir), str(self.job_id))

        self.output_directory = os.path.join(dest_dir, "output")
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory)

        # create input file name
        input_string = str(self.experiment_name) + '_' + str(self.job_id) + '.dat'
        self.input_file = os.path.join(dest_dir, input_string)

        # create output file name
        output_string = str(self.experiment_name) + '_' + str(self.job_id) + '.out'
        self.output_file = os.path.join(self.output_directory, output_string)

    def run_job(self):
        """ Actual method to run the job on computing machine
            using run_subprocess method from base class
        """
        # assemble run command
        command_string = self.assemble_command_string(ansys_variant='v15_linux')

        _, stderr, self.pid = self.run_subprocess(command_string)
        if stderr:
            self.result = None  # This is necessary to detect failed jobs
            self.job['status'] = 'failed'

    def assemble_command_string(self, ansys_variant):
        """  Assemble command list

            Args:
                ansys_variant (str): Set ANSYS version and OS

            Returns:
                list: command list to execute ANSYS

        """
        command_list = []
        if ansys_variant == 'v15_linux':
            command_list = [
                self.main_executable,
                "-b -g -p aa_t_a -dir ",
                self.output_directory,
                "-i ",
                self.input_file,
                "-j ",
                str(self.experiment_name) + '_' + str(self.job_id),
                "-s read -l en-us -t -d X11 > ",
                self.output_file
            ]
        elif ansys_variant == 'v19_windows':
            command_list = [
                self.main_executable,
                "-p ansys -smp -np 1 -lch -dir",
                self.output_directory,
                "-j",
                str(self.experiment_name) + '_' + str(self.job_id),
                "-s read -l en-us -b -i",
                self.input_file,
                "-o",
                self.output_file,
                "-custom",
                self.custom_executable,
            ]
        else:
            raise RuntimeError("Unknown ANSYS Varaint, fix your config file")

        return ' '.join(filter(None, command_list))
