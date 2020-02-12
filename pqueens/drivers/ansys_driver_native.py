import os
import subprocess
import importlib.util
from pqueens.utils.injector import inject


def ansys_driver_native(job):
    """
        Driver to run ANSYS natively on host machine

        Args:
            job (dict): Dict containing all information to run the simulation

        Returns:
            float: result
    """

    sys.stderr.write("Running ANSYS job.\n")

    # Add directory to the system path.
    sys.path.append(os.path.realpath(job['expt_dir']))

    # Change into the directory.
    os.chdir(job['expt_dir'])
    sys.stderr.write("Changed into dir %s\n" % (os.getcwd()))

    # get params dict
    params = job['params']
    driver_params = job['driver_params']

    # assemble input file name
    ansys_input_file = job['expt_dir'] + '/' + job['expt_name'] + '_' + str(job['id']) + '.inp'
    ansys_output_file = job['expt_dir'] + '/' + job['expt_name'] + '_' + str(job['id'])

    sys.stderr.write("ansys_input_file %s\n" % ansys_input_file)

    # create input file using injector
    inject(params, driver_params['input_template'], ansys_input_file)
    # second injection to set output file
    ansys_output = {}
    ansys_output["output"] = ansys_output_file
    inject(ansys_output, ansys_input_file, ansys_input_file)

    # get ansys run and post process command
    ansys_cmd = [
        driver_params['path_to_executable'],
        "-b -g -p aa_t_a -dir ",
        job['expt_dir'],
        "-i ",
        ansys_input_file,
        "-j ",
        job['expt_name'] + "_" + str(job["id"]),
        "-s read -l en-us -t -d X11 > ",
        ansys_output_file,
    ]

    # run ansys
    p = subprocess.Popen(ansys_cmd)
    temp_out = p.communicate()
    print(temp_out)

    result = None

    # call post post process script to extract result from monitor file
    spec = importlib.util.spec_from_file_location("module.name", driver_params['post_post_script'])
    post_post_proc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(post_post_proc)
    result = post_post_proc.run(ansys_output_file + '.out')

    sys.stderr.write("Got result %s\n" % (result))

    return result
