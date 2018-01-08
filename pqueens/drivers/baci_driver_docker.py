import sys
import os
import importlib.util
from pqueens.utils.injector import inject
import docker

def baci_driver_docker(job):
    """
        Driver to run BACI simulation inside Docker container

        Args:
            job(dict): Dict containing all information to run the simulation

        Returns:
            (float): result
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

    # assemble baci run and post process command
    baci_cmd = driver_params['path_to_executable'] + ' ' + baci_input_file + ' ' + baci_output_file
    post_cmd = driver_params['path_to_postprocessor'] + ' ' + driver_params['post_process_options'] + ' --file='+baci_output_file

    volume_map = {job['expt_dir']: {'bind': job['expt_dir'], 'mode': 'rw'}}

    # run BACI in container
    temp_out = run_baci(driver_params['docker_container'], baci_cmd, volume_map)

    temp_out = run_post_processing(driver_params['docker_container'], post_cmd,
                                   volume_map)

    print(temp_out)

    result = run_post_post_processing(driver_params['post_post_script'],
                                      baci_output_file)

    sys.stderr.write("Got result %s\n" % (result))

    return result


def run_baci(container_name, baci_cmd, volume_map):
    """ Run BACI inside docker container

    Args:
        container_name (string): Name of container to run
        baci_cmd (string):       Command to run BACI
        volume_map (string):     Define which folders get mapped into container

    Returns:
        string: terminal output
    """
    client = docker.from_env()
    temp_out = client.containers.run(container_name, baci_cmd, volumes=volume_map)
    return temp_out

def run_post_processing(container_name, post_cmd, volume_map):
    """ Run post processing inside docker container

    Args:
        container_name (string): Name of container to run
        post_cmd (string):       Command to run post processing
        volume_map (string):     Define which folders get mapped into container

    Returns:
        string: terminal output
    """
    client = docker.from_env()
    temp_out = client.containers.run(container_name, post_cmd, volumes=volume_map)
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
    result = post_post_proc.run(baci_output_file+'.mon')
    return result
