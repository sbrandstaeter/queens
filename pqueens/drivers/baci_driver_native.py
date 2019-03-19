import sys
import os
import subprocess
import importlib.util
from pqueens.utils.injector import inject

def baci_driver_native(job):
    """
        Driver to run BACI natively on host machine

        Args:
            job (dict): Dict containing all information to run the simulation

        Returns:
            float: result
    """

    sys.stderr.write("Running BACI job.\n")

    # Add directory to the system path.
    sys.path.append(os.path.realpath(job['expt_dir']))

    # Change into the directory.
    os.chdir(job['expt_dir'])
    sys.stderr.write("Changed into dir %s\n" % (os.getcwd()))

    # get params dict
    params = job['params']
    driver_params = job['driver_params']

    # assemble input file name
    baci_input_file = job['expt_dir'] + '/' + job['expt_name'] + '_' + str(job['id']) + '.dat'
    baci_output_file = job['expt_dir'] + '/'+ job['expt_name'] + '_' + str(job['id'])

    sys.stderr.write("baci_input_file %s\n" % baci_input_file)

    # create input file using injector
    inject(params, driver_params['input_template'], baci_input_file)

    # assemble baci run command
    baci_cmd = driver_params['path_to_executable'] + ' ' + baci_input_file + ' ' + baci_output_file

    # run BACI
    temp_out = run_baci(baci_cmd)
    print("Communicate run baci")
    print(temp_out)

    # Post-process BACI run
    for i, post_process_option in enumerate(driver_params['post_process_options']):
        post_cmd = driver_params['path_to_postprocessor'] + ' ' + post_process_option + ' --file='+baci_output_file + ' --output='+baci_output_file+'_'+str(i+1)
        temp_out = run_post_processing(post_cmd)
        print("Communicate post-processing")
        print(temp_out)

    # Call post post-processing script
    result = run_post_post_processing(driver_params['post_post_script'],
                                      baci_output_file)

    return result


def run_baci(baci_cmd):
    """ Run BACI via subprocess

    Args:
        baci_cmd (string):       Command to run BACI

    Returns:
        string: terminal output
    """
    p = subprocess.Popen(baci_cmd,
                         shell=True)
    temp_out = p.communicate()

    return temp_out


def run_post_processing(post_cmd):
    """ Run BACI post processor via subprocess

    Args:
        post_cmd (string):       Command to run post processing

    Returns:
        string: terminal output
    """

    p = subprocess.Popen(post_cmd,
                         shell=True)
    temp_out = p.communicate()

    return temp_out


def run_post_post_processing(post_post_script, baci_output_file):
    """ Run script to extract results from monitor file

    Args:
        post_post_script (string): name of script to run
        baci_output_file (string): name of file to use

    Returns:
        float: actual simulation result
    """
    # call post post process script to extract result from monitor file
    spec = importlib.util.spec_from_file_location("module.name", post_post_script)
    post_post_proc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(post_post_proc)
    result = post_post_proc.run(baci_output_file)

    sys.stderr.write("Got result %s\n" % (result))

    return result

