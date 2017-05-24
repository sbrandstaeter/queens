import sys
import os
from pqueens.utils.injector import inject
import numpy as np
import docker

def baci_driver(job):
    """
        Driver to run BACI simulation inside Docker container

        Args:
            job(dict): Dict containing all information to run the simulation

        Returns:
            (float): result
    """

    # get docker client
    client = docker.from_env()

    # Runnin a BACI job
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

    output_filename = os.path.join(job['expt_dir'], '%08dbaci.out' % job['id'])
    output_file = open(output_filename, 'w')

    # create input file using injector
    inject(params,driver_params['input_template'],baci_input_file)

    # run BACI in container
    # assemble shell commands
    volume_map = {job['expt_dir']: {'bind': job['expt_dir'], 'mode': 'rw'}}
    baci_cmd = './baci-release ' + baci_input_file + ' ' + baci_output_file
    post_cmd = './post_drt_monitor' + ' --field=structure --file='+baci_output_file + ' --node=23 --start=25'

    temp_out = client.containers.run("adco/baci-baked-in-centos", baci_cmd,volumes=volume_map)
    #sys.stderr.write('container output %s' temp_out)
    temp_out = client.containers.run("adco/baci-baked-in-centos", post_cmd,volumes=volume_map)
    #sys.stderr.write('container output %s' temp_out)
    print(temp_out)

    # read in resutls
    line = np.loadtxt(baci_output_file+'.mon', comments="#",skiprows=4, unpack=False)
    result = np.sqrt(line[1]**2+line[2]**2+line[3]**2)

    # Change back out.
    os.chdir('..')
    #
    sys.stderr.write("Got result %s\n" % (result))

    return result
