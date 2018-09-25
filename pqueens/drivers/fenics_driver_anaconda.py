import sys
import os
import subprocess
import importlib.util
from pqueens.utils.injector import inject

def fenics_driver_anaconda(job):
    """
        Driver to run FENICS natively on host machine

        Args:
            job (dict): Dict containing all information to run the simulation

        Returns:
            float: result
    """

    sys.stderr.write("Running FENICS job.\n")

    # Add directory to the system path.
    sys.path.append(os.path.realpath(job['expt_dir']))

    # Change into the directory.
    os.chdir(job['expt_dir'])
    sys.stderr.write("Changed into dir %s\n" % (os.getcwd()))

    # get params dict
    params = job['params']
    driver_params = job['driver_params']

    # assemble input file name
    fenics_input_file = job['expt_dir'] + '/' + job['expt_name'] + '_' + str(job['id']) + '.dat'
    fenics_output_file = job['expt_dir'] + '/'+ job['expt_name'] + '_' + str(job['id'])

    sys.stderr.write("fenics_input_file %s\n" % fenics_input_file)

    # create input file using injector
    inject(params, driver_params['input_template'], fenics_input_file)

    # get fenics run and post process command
    fenics_cmd = [driver_params['path_to_executable'], fenics_input_file, fenics_output_file]
    post_cmd = [driver_params['path_to_postprocessor'],'--file='+fenics_output_file]
    post_cmd = post_cmd + driver_params['post_process_options']

    # run FENICS
    p = subprocess.Popen(fenics_cmd)
    temp_out = p.communicate()
    print(temp_out)

    p = subprocess.Popen(post_cmd)
    temp_out = p.communicate()
    print(temp_out)

    # call post post process script to extract result from monitor file
    spec = importlib.util.spec_from_file_location("module.name", driver_params['post_post_script'])
    post_post_proc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(post_post_proc)
    result = post_post_proc.run(fenics_output_file+'.mon')

    sys.stderr.write("Got result %s\n" % (result))

    return result
