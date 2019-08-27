import os
import json
import sys
import importlib.util

from pqueens.drivers.driver import Driver
from pqueens.database.mongodb import MongoDB


class Baci_driver_schmarrn(Driver): #TODO needs to be adjusted!!!
    """ Driver to run BACI on the HPC cluster schmarrn (via PBS/Torque)

    Args:

    Returns:
    """
    def __init__(self, base_settings):
        super(Baci_driver_schmarrn, self).__init__(base_settings)

        self.port = base_settings['port']
        self.num_procs = base_settings['num_procs']
        address ='10.10.0.1:'+ str(self.port) #TODO change to linux command to find master node
        self.database = MongoDB(database_address=address)# TODO we assume that 10.10.0.1 is always the master node for slurm

    @classmethod
    def from_config_create_driver(cls, config, base_settings):
        """ Create Driver from JSON input file

        Args:

        Returns:
            driver: Baci_driver_schmarrn object
        """
        base_settings['experiment_name'] =config['experiment_name']
        return cls(base_settings)


    def setup_mpi(self,ntasks):
        """ setup MPI environment

            Args:
                num_procs (int): Number of processors to use

            Returns:
                str, str: MPI runcommand, MPI flags
        """
#        srcdir = os.environ["PBS_O_WORKDIR"]
#        os.chdir(srcdir)
#        address ='10.10.0.1:'+ self.port
#        self.database(database_address=address)# TODO we assume that 10.10.0.1 is always the master node for slurm
#        mpi_run = '/opt/openmpi/1.6.2/gcc48/bin/mpirun'
#        mpi_home = '/opt/openmpi/1.6.2/gcc48'
#
#        os.environ["MPI_HOME"] = mpi_home
#        os.environ["MPI_RUN"] = mpi_run
#        os.environ["LD_LIBRARY_PATH"] = mpi_home #TODO seemed to be empty thats we just add mpi_home here, nothing to append; might change lateron
#
#        # Add non-standard shared library paths
#        my_env = os.environ.copy()
        # determine 'optimal' flags for the problem size
        #num_procs = self.mpi_config['num_procs']
        if ntasks%16 == 0:
            self.mpi_flags = "--mca btl openib,sm,self --mca mpi_paffinity_alone 1"
        else:
            self.mpi_flags = "--mca btl openib,sm,self"

#        self.mpi_config['mpi_run'] = mpi_run
#        self.mpi_config['my_env'] = my_env
#        self.mpi_conig['flags'] = mpi_flags
#
#        # determine number of processors from nodefile
#        pbs_nodefile = os.environ["PBS_NODEFILE"]
#        command_list = ['cat', pbs_nodefile, '|', 'wc', '-l']
#        command_string = ' '.join(command_list)
#        self.mpi_config['nodelist_procs'] = int(self.run_subprocess(command_string))
