import os
import json
import sys
import importlib.util
from pqueens.database.mongodb import MongoDB
from pqueens.drivers.driver import Driver

class Baci_driver_bruteforce(Driver):
    """ Driver to run BACI on the HPC cluster bruteforce (via Slurm)

    Args:

    Returns:
    """
    def __init__(self, base_settings):
        super(Baci_driver_bruteforce, self).__init__(base_settings)
        port = base_settings['port']
        address ='10.10.0.1:'+ str(port) #TODO change to linux command to find master node
        self.database = MongoDB(database_address=address)

    @classmethod
    def from_config_create_driver(cls, config, base_settings):
        """ Create Driver from JSON input file

        Args:

        Returns:
            driver: Baci_driver_bruteforce object
        """
        return cls(base_settings)


    def setup_mpi(self, ntasks):
        """ setup MPI environment

            Args:
                num_procs (int): Number of processors to use

            Returns:
                str, str: MPI runcommand, MPI flags
        """
        if ntasks%16 == 0:
            self.mpi_flags = "--mca btl openib,sm,self --mca mpi_paffinity_alone 1"
        else:
            self.mpi_flags = "--mca btl openib,sm,self"
