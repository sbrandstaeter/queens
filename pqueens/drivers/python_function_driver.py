"""Python function module.

This module is outdated and should therefore not be used.
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)


def python_function_driver(job):
    """Driver for python function specified in job dictionary.

    Args:
        job (dict): Dictionary containing function

    Returns:
        result (float): Result of run
    """
    # run Python function
    _logger.info("Running job for Python function.\n")

    # add directory to system path.
    sys.path.append(str(Path(job['experiment_dir'])))

    # change to directory.
    os.chdir(job['experiment_dir'])
    _logger.info("Changed to directory %s\n", Path.cwd())

    # convert dict to vector of parameters.
    params = {}
    for name, param in job['params'].items():
        vals = param['values']
        if param['type'].lower() == 'float':
            params[name] = np.array(vals)
        elif param['type'].lower() == 'int':
            params[name] = np.array(vals, dtype=int)
        elif param['type'].lower() == 'enum':
            params[name] = vals
        else:
            raise TypeError("Unknown parameter type.")

    # load module and run
    main_file = job['main-file']
    if main_file[-3:] == '.py':
        main_file = main_file[:-3]
    _logger.info('Importing %s.py\n', main_file)
    module = __import__(main_file)
    _logger.info('Running %s.main()\n', main_file)
    result = module.main(job['id'], params)

    # change back
    os.chdir('..')

    _logger.info("Got result %s\n", result)

    return result
