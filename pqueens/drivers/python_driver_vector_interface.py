import sys
import os
import numpy as np


def python_driver_vector_interface(job):
    """
        Driver to call a python function specified in the job dict

        Args:
            job (dict): Dict containing all information to run the function

        Returns:
            (float): result
    """
    # Run a Python function
    sys.stderr.write("Running python job.\n")

    # Add directory to the system path.
    sys.path.append(os.path.realpath(job['expt_dir']))

    # Change into the directory.
    os.chdir(job['expt_dir'])
    sys.stderr.write("Changed into dir %s\n" % (os.getcwd()))

    # Convert dict into vector of parameters.
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

    # Load up this module and run
    main_file = job['driver_params']['main-file']
    if main_file[-3:] == '.py':
        main_file = main_file[:-3]
    sys.stderr.write('Importing %s.py\n' % main_file)
    module = __import__(main_file)
    sys.stderr.write('Running %s.main()\n' % main_file)
    result = module.main(job['id'], params)

    # Change back out.
    os.chdir('..')

    sys.stderr.write("Got result %s\n" % (result))

    return result
