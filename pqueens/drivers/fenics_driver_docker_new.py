import sys
import os
from pqueens.utils.injector import inject
import docker

def fenics_driver_docker(job):
    """
        Driver to run FENICS simulation inside Docker container

        Args:
            job(dict): Dict containing all information to run the simulation

        Returns:
            (float): result
    """

    # get docker client
    client = docker.from_env()
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
    fenics_input_file = job['expt_dir'] + '/' + job['expt_name'] + '_' + str(job['id']) + '.py'
    fenics_output_file = job['expt_dir'] + '/'+ job['expt_name'] + '_' + str(job['id']) + '.out'

    sys.stderr.write("fenics_input_file %s\n" % fenics_input_file)

    # create input file using injector
    inject(params, driver_params['input_template'], fenics_input_file)

    # get fenics run and post process command
    fenics_cmd = fenics_input_file + ' --output_file='+fenics_output_file

    #  setup volume map
    volume_map = {job['expt_dir']: {'bind': job['expt_dir'], 'mode': 'rw'},
                  'instant-cache' : {'bind' : '/home/fenics/.instant', 'mode': 'rw'}}

    # run fenics in docker container
    temp_out = client.containers.run(driver_params['docker_container'],
                                     fenics_cmd, entrypoint='python3',
                                     stdout=True, stderr=True, volumes=volume_map)


    print(temp_out)

    my_file = open(fenics_output_file, 'r')
    result = my_file.readline()

    sys.stderr.write("Got result %s\n" % (result))

    return result
