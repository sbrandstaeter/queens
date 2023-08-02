# -*- coding: utf-8 -*-
"""Drivers.

This package contains a set of driver scripts, which are used to make
the actual call to the simulation software.
"""
from pqueens.utils.import_utils import get_module_class

VALID_TYPES = {
    'mpi': ["pqueens.drivers.mpi_driver", "MpiDriver"],
    'jobscript': ["pqueens.drivers.jobscript_driver", "JobscriptDriver"],
}
