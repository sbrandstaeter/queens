# -*- coding: utf-8 -*-
"""Drivers.

This package contains a set of driver scripts, which are used to make
the actual call to the simulation software.
"""

from queens.drivers.jobscript_driver import JobscriptDriver
from queens.drivers.mpi_driver import MpiDriver

VALID_TYPES = {
    "mpi": MpiDriver,
    "jobscript": JobscriptDriver,
}
