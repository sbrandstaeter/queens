# -*- coding: utf-8 -*-
"""Drivers.

This package contains a set of driver scripts, which are used to make
the actual call to the simulation software.
"""

VALID_TYPES = {
    'mpi': ["queens.drivers.mpi_driver", "MpiDriver"],
    'jobscript': ["queens.drivers.jobscript_driver", "JobscriptDriver"],
}
