"""Python function module.

This module is outdated and should therefore not be used.
"""
import logging
import os
import sys

import numpy as np

_logger = logging.getLogger(__name__)


def python_function_driver(job):
    """Driver for python function specified in job dictionary.

    Args:
        job (dict): dictionary containing function

    Returns:
        result (float): result of run
    """
    # run Python function
    _logger.info("Running job for Python function.\n")

    # add directory to system path.
    sys.path.append(os.path.realpath(job['experiment_dir']))

    # change to directory.
    os.chdir(job['experiment_dir'])
    _logger.info("Changed to directory %s\n", os.getcwd())

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
            raise Exception("Unknown parameter type.")

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
