""" This should be a docstring """

import re
import os
from pqueens.drivers.driver import Driver


class BaciDriverBruteforce(Driver):
    """ Driver to run BACI on the HPC cluster bruteforce (via Slurm)

    Args:

    Returns:
    """

    def __init__(self, base_settings):
        super(BaciDriverBruteforce, self).__init__(base_settings)

    @classmethod
    def from_config_create_driver(cls, config, base_settings):
        """ Create Driver from JSON input file

        Args:

        Returns:
            driver: Baci_driver_bruteforce object
        """
        base_settings['address'] = '10.10.0.1:' + str(base_settings['port'])
        # TODO change to linux command to find master node
        base_settings['experiment_name'] = config['experiment_name']
        return cls(base_settings)

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
        self.input_file = (
            dest_dir + '/' + str(self.experiment_name) + '_' + str(self.job_id) + '.dat'
        )

        # create output file name
        self.output_file = (
            output_directory + '/' + str(self.experiment_name) + '_' + str(self.job_id)
        )
        self.output_scratch = self.experiment_name + '_' + str(self.job_id)

    def run_job(self):
        """ Actual method to run the job on computing machine
            using run_subprocess method from base class
        """
        # assemble run command
        command_list = [
            self.executable,
            self.input_file,
            self.output_scratch,
        ]  # This is already within pbs
        # Here we call directly the executable inside the container not the jobscript!
        command_string = ' '.join(filter(None, command_list))
        stdout, stderr, self.pid = self.run_subprocess(command_string)

        if stderr:  # TODO this will not work yet for bruteforce
            # but a similar solution should be tested and implemented for bruteforce
            if re.fullmatch(
                r'/bin/sh: line 0: cd: /scratch/PBS_\d+.master.cluster: No such file or directory\n',
                stderr,
            ):
                pass
            else:
                self.result = None  # This is necessary to detect failed jobs
                self.job['status'] = 'failed'
