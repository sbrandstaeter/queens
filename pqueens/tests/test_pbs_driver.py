'''
Created on February  4th 2018
@author: jbi

'''
import argparse
import unittest
import mock

from pqueens.drivers.dummy_driver_baci_pbs_kaiser import main as driver_main
from pqueens.drivers.dummy_driver_baci_pbs_kaiser import setup_mpi


class TestPBSDriver(unittest.TestCase):

    def test_setup_mpi(self):
        mpi_run, mpi_flags = setup_mpi(16)
        self.assertEqual(mpi_flags, "--mca btl openib,sm,self --mca mpi_paffinity_alone 1")
        self.assertEqual(mpi_run, "/opt/openmpi/1.6.2/gcc48/bin/mpirun")
        mpi_run, mpi_flags = setup_mpi(8)
        self.assertEqual(mpi_flags, "--mca btl openib,sm,self")
