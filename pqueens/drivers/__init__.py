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


def from_config_create_driver(config, driver_name):
    """Create driver from problem description.

    Args:
        config (dict):  Dictionary containing configuration from QUEENS input file
        driver_name (str): Name of driver

    Returns:
        driver (obj): Driver object
    """
    driver_options = config[driver_name]
    driver_class = get_module_class(driver_options, VALID_TYPES, "type")
    driver = driver_class.from_config_create_driver(config=config, driver_name=driver_name)
    return driver
