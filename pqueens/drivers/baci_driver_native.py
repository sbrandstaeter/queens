""" This should be a docstring """

import os
from pqueens.drivers.driver import Driver


class BaciDriverNative(Driver):
    """ Driver to run BACI natively on workstation

        Args:
            job (dict): Dict containing all information to run the simulation

        Returns:
            float: result
    """
    def __init__(self, base_settings):
        super(BaciDriverNative, self).__init__(base_settings)
        self.mpi_config = {}

    @classmethod
    def from_config_create_driver(cls, config, base_settings):
        """ Create Driver from input file

        Args:
        Returns:
            driver: BaciDriverNative object

        """
        base_settings['address'] = 'localhost:27017'
        return cls(base_settings)

# ----------------- CHILD METHODS THAT NEED TO BE IMPLEMENTED -----------------
    def setup_dirs_and_files(self):
        """ Setup directory structure

            Args:
                driver_options (dict): Options dictionary

            Returns:
                str, str, str: simualtion prefix, name of input file, name of output file
        """
        # base directories
        dest_dir = str(self.experiment_dir) + '/' + str(self.job_id)

        # Depending on the input file, directories will be created locally or on a cluster
        output_directory = os.path.join(dest_dir, 'output')
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        # create input file name
        self.input_file = dest_dir + '/' + str(self.experiment_name) +\
                                     '_' + str(self.job_id) + '.dat'

        # create output file name
        self.output_file = output_directory + '/' + str(self.experiment_name) +\
                                              '_' + str(self.job_id)

    def run_job(self):
        """ Actual method to run the job on computing machine
            using run_subprocess method from base class
        """
        # assemble run command
        self.setup_mpi(self.num_procs)
        command_list = ['mpirun', '-np', str(self.num_procs), self.mpi_flags, self.executable,
                        self.input_file, self.output_file]

        command_string = ' '.join(filter(None, command_list))
        stdout, stderr, self.pid = self.run_subprocess(command_string)

        if stderr != "":
            raise RuntimeError(stderr+stdout)

    def setup_mpi(self, num_procs):  # TODO this is not needed atm
        pass
