""" This should be a docstring """

import re
import os
from pqueens.drivers.driver import Driver


class BaciDriverBruteforce(Driver):
    """ Driver to run BACI on the HPC cluster bruteforce (via Slurm)

    Args:

    Returns:
    """

    def __init__(self, base_settings, workdir):
        super(BaciDriverBruteforce, self).__init__(base_settings)
        self.workdir = workdir

    @classmethod
    def from_config_create_driver(cls, config, base_settings, workdir):
        """ Create Driver from JSON input file

        Args:

        Returns:
            driver: Baci_driver_bruteforce object
        """
        base_settings['address'] = '10.10.0.1:' + str(base_settings['port'])
        # TODO change to linux command to find master node
        base_settings['experiment_name'] = config['experiment_name']
        return cls(base_settings, workdir)

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
            'cd',
            self.workdir,
            r'&&',
            self.executable,
            self.input_file,
            self.output_scratch,
        ]  # This is already within pbs
        # Here we call directly the executable inside the container not the jobscript!
        command_string = ' '.join(filter(None, command_list))
        # Call BACI
        stdout, stderr, self.pid = self.run_subprocess(command_string)
        # Print the stderr of BACI call to slurm file (SLURM_{SLURM_ID}.txt)
        print(stderr)
        # Print the stdout of BACI call to slurm file (SLURM_{SLURM_ID}.txt)
        print(stdout)
        # Print the stderr of BACI to a separate file in the output directory
        with open(self.output_file + "_BACI_stderr.txt", "a") as text_file:
            print(stderr, file=text_file)
        # Print the stdout of BACI to a separate file in the output directory
        with open(self.output_file + "_BACI_stdout.txt", "a") as text_file:
            print(stdout, file=text_file)

        if stderr:
            # TODO: fix this hack
            # For the second call of remote_main.py with the --post=true flag
            # (see the jobscript_slurm_queens.sh), the workdir does not exist anymore.
            # Therefore, change directory in command_list ("cd self.workdir") does throw an error.
            # We catch this error to detect that we are in a postprocessing call of the driver.
            if re.fullmatch(
                r'/bin/sh: line 0: cd: /scratch/SLURM_\d+: No such file or directory\n', stderr
            ):
                pass
            else:
                self.result = None  # This is necessary to detect failed jobs
                self.job['status'] = 'failed'
