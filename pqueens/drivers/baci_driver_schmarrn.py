""" This should be a docstring """

import os
from pqueens.drivers.driver import Driver


class BaciDriverSchmarrn(Driver):
    """ Driver to run BACI on the HPC cluster schmarrn (via PBS/Torque)

    Args:

    Returns:
    """
    def __init__(self, base_settings):
        super(BaciDriverSchmarrn, self).__init__(base_settings)

        self.num_procs = base_settings['num_procs']

    @classmethod
    def from_config_create_driver(cls, config, base_settings):
        """ Create Driver from JSON input file

        Args:

        Returns:
            driver: BaciDriverSchmarrn object
        """
        port = base_settings['port']
        base_settings['address'] = '10.10.0.1:' + str(port)
        return cls(base_settings)

# ----------------- CHILD METHODS THAT NEED TO BE IMPLEMENTED -----------------
    def setup_mpi(self, ntasks):
        """ setup MPI environment

            Args:
                num_procs (int): Number of processors to use

            Returns:
                str, str: MPI runcommand, MPI flags
        """
        if ntasks % 16 == 0:
            self.mpi_flags = "--mca btl openib,sm,self --mca mpi_paffinity_alone 1"
        else:
            self.mpi_flags = "--mca btl openib,sm,self"

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
