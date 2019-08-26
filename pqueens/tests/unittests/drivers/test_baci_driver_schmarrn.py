'''
Created on February  4th 2018
@author: jbi

'''
import pytest

from pqueens.database.mongodb import MongoDB
from pqueens.drivers.baci_driver_schmarrn import BaciDriverSchmarrn


@pytest.fixture()
def baci_driver_schmarrn(driver_base_settings, mocker):
    mocker.patch.object(MongoDB, '__init__', return_value=None)
    return  BaciDriverSchmarrn(base_settings=driver_base_settings)


def test_setup_mpi(baci_driver_schmarrn):
    baci_driver_schmarrn.setup_mpi(16)
    assert baci_driver_schmarrn.mpi_flags == "--mca btl openib,sm,self --mca mpi_paffinity_alone 1"
    baci_driver_schmarrn.setup_mpi(8)
    assert baci_driver_schmarrn.mpi_flags == "--mca btl openib,sm,self"
