import os
import json
import sys
import importlib.util

from .drivers import Driver


class Baci_driver_bruteforce(Driver):
    """ Drover to run BACI on the HPC cluster bruteforce (via Slurm)

    Args:

    Returns:
    """
    def __init__(self, base_settings)
        super(Baci_driver_bruteforce, self).__init__(base_settings)

    @classmethod
    def from_config_create_driver(cls, config, base_settings)
        """ Create Driver from JSON input file

        Args:

        Returns:
            driver: Baci_driver_bruteforce object
        """
        num_procs = config[]
        return cls(base_settings)


    def setup_mpi(self):
    """ setup MPI environment

            Args:
                num_procs (int): Number of processors to use

            Returns:
                str, str: MPI runcommand, MPI flags
        """
        srcdir = os.environ["SLURM_SUBMIT_DIR"]
        os.chdir(srcdir)
        self.database(database_address='10.10.0.1:27017')# TODO This is probabily worng here and should be executed immediately; also hard coded port should be removed!
        mpi_run = '/cluster/mpi/intel/openmpi/1.10.1/bin/mpirun'
        mpi_home = '/cluster/mpi/intel/openmpi/1.10.1'

        os.environ["MPI_HOME"] = mpi_home
        os.environ["MPI_RUN"] = mpi_run
        os.environ["LD_LIBRARY_PATH"] += "/cluster/mpi/intel/openmpi/1.10.1'/lib"

        # Add non-standard shared library paths
        my_env = os.environ.copy()
        # determine 'optimal' flags for the problem size
        num_procs = self.mpi_config['num_procs']
        if num_procs%16 == 0:
            mpi_flags = "--mca btl openib,sm,self --mca mpi_paffinity_alone 1"
        else:
            mpi_flags = "--mca btl openib,sm,self"

        self.mpi_config['mpi_run'] = mpi_run
        self.mpi_config['my_env'] = my_env
        self.mpi_conig['flags'] = mpi_flags

        # determine number of processors from nodefile
        slurm_nodefile = os.environ["SLURM_JOB_NODELIST"]
        command_list = ['cat', slurm_nodefile, '|', 'wc', '-l']
        command_string = ' '.join(command_list)
        self.mpi_config['nodelist_procs'] = int(self.run_subprocess(command_string))
        #TODO : maybe better return a complete mpi_command instead of dict?
