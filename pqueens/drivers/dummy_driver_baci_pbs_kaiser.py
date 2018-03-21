import os
import subprocess
import json
import sys
import time
from pqueens.database.mongodb import MongoDB
from pqueens.utils.injector import inject
import numpy as np

def main(args):
    """
        Very basic lauchner script to launch BACI jobs on Kaiser cluster.

        Attention a lot of things are hard coded here that probably should not be
        hard coded, so proceed with caution.

        Args:
            args (JSON document): file-like object containing a JSON document
    """

    # all necessary information is passed via this dictionary
    driver_options = json.loads(args)

    # get PBS working directory
    srcdir = os.environ["PBS_O_WORKDIR"]
    os.chdir(srcdir)

    # connect to database and get job parameters
    db = MongoDB(database_address=driver_options['database_address'])

    job = init_job(driver_options, db)

    _, baci_input_file, baci_output = setup_dirs_and_files(driver_options)

    # create actual input file in experiment dir folder
    inject(job['params'], driver_options['input_template'], baci_input_file)

    # assemble command to run BACI
    runcommand_string = get_runcommand_string(driver_options, baci_input_file, baci_output)

    #run BACI
    run(runcommand_string)

    # assemble command to run post processor
    postcommand_string = get_postcommand_string(driver_options, baci_output)

    # run postprocessing
    run(postcommand_string)

    result = do_dummy_postpostprocessing(baci_output)

    finish_job(driver_options, db, job, result)

def get_num_nodes():
    """ determine number of processors from nodefile """
    pbs_nodefile = os.environ["PBS_NODEFILE"]
    #print(pbs_nodefile)
    command_list = ['cat', pbs_nodefile, '|', 'wc', '-l']
    command_string = ' '.join(command_list)
    p = subprocess.Popen(command_string,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         shell=True,
                         universal_newlines=True)
    procs, _ = p.communicate()
    return int(procs)

def setup_mpi(num_procs):
    """ setup MPI environment

        Args:
            num_procs (int): Number of processors to use

        Returns:
            str, str: MPI runcommand, MPI flags
    """
    mpi_run = '/opt/openmpi/1.6.2/gcc48/bin/mpirun'
    mpi_home = '/opt/openmpi/1.6.2/gcc48'

    os.environ["MPI_HOME"] = mpi_home
    os.environ["MPI_RUN"] = mpi_run

    # Add non-standard shared library paths
    # "LD_LIBRARY_PATH" seems to be also empty, so simply set it to MPI_HOME
    # eventually this should changed to mereyl append the MPI_HOME path
    os.environ["LD_LIBRARY_PATH"] = mpi_home

    # determine 'optimal' flags for the problem size
    if num_procs%16 == 0:
        mpi_flags = "--mca btl openib,sm,self --mca mpi_paffinity_alone 1"
    else:
        mpi_flags = "--mca btl openib,sm,self"

    return mpi_run, mpi_flags

def setup_dirs_and_files(driver_options):
    """ Setup directory structure

        Args:
            driver_options (dict): Options dictionary

        Returns:
            str, str, str: simualtion prefix, name of input file, name of output file
    """
    dest_dir = str(driver_options['experiment_dir']) + '/' + \
              str(driver_options['job_id'])

    prefix = str(driver_options['experiment_name']) + '_' + \
             str(driver_options['job_id'])

    output_directory = os.path.join(dest_dir, 'output')
    if not os.path.isdir(output_directory):
        # make complete directory tree
        os.makedirs(output_directory)

    # create input file using injector
    baci_input_file = dest_dir + '/' + str(driver_options['experiment_name']) + \
                      '_' + str(driver_options['job_id']) + '.dat'

    # create ouput file name
    baci_output = output_directory + '/' + str(driver_options['experiment_name']) + \
                      '_' + str(driver_options['job_id'])

    return prefix, baci_input_file, baci_output

def init_job(driver_options, db):
    """ Initialize job in database

        Args:
            driver_options (dict): Options dictionary
            db (MongoDB) :         MongoDB object

        Returns:
            dict: Dictionary with job information

    """
    job = db.load(driver_options['experiment_name'], driver_options['batch'], 'jobs',
                  {'id' : driver_options['job_id']})

    start_time = time.time()
    job['start time'] = start_time

    db.save(job, driver_options['experiment_name'], 'jobs', driver_options['batch'],
            {'id' : driver_options['job_id']})

    sys.stderr.write("Job launching after %0.2f seconds in submission.\n"
                     % (start_time-job['submit time']))
    return job

def finish_job(driver_options, db, job, result):
    """ Change status of job to completed in database

        Args:
            driver_options (dict):    Options dictionary
            db             (MongoDB): MongoDB object
            job            (dict):    Dictionary with job information
            result         (float):   Result of simulation associated with job
    """
    end_time = time.time()

    job['result'] = result
    job['status'] = 'complete'
    job['end time'] = end_time

    db.save(job, driver_options['experiment_name'], 'jobs', driver_options['batch'],
            {'id' : driver_options['job_id']})

def do_dummy_postpostprocessing(baci_output):
    """ Execute dummy post post processing step

        Args:
            baci_output (str): Path to BACI monitor file

        Returns:
            float: Postprocessed result
    """
    line = np.loadtxt(baci_output+'.mon', comments="#", skiprows=4, unpack=False)
    # for now simply compute norm of displacement
    result = np.sqrt(line[1]**2+line[2]**2+line[3]**2)
    print('And the results is: {}'.format(result))
    print('Written result to database')
    return result

def get_runcommand_string(driver_options, baci_input_file, baci_output):
    """ Assemble run command for BACI

        Args:
            driver_options (dict): Options dictionary
            baci_input_file (str): String with baci_input_file
            baci_output     (str): String with output file/path

        Returns:
            str: Complete command to execute BACI
    """
    procs = get_num_nodes()
    mpir_run, mpi_flags = setup_mpi(procs)
    executable = driver_options['executable']

    # note that we directly write the output to the home folder and do not create
    # the appropriate directories on the nodes. This should be changed at some point.
    # So long be careful !

    runcommand_list = [mpir_run, mpi_flags, '-np', str(procs), executable,
                       baci_input_file, baci_output]
    runcommand_string = ' '.join(runcommand_list)
    return runcommand_string


def get_postcommand_string(driver_options, baci_output):
    """ Assemble post processing command for BACI

        Args:
            driver_options (dict): Options dictionary
            baci_output (str):     Path to BACI output file

        Returns:
            str: Post processing command for BACI
    """
    procs = get_num_nodes()
    mpir_run, mpi_flags = setup_mpi(procs)
    post_processor_exec = driver_options['post_processor']
    monitor_file = '--file=' + str(baci_output)
    post_process_command = driver_options['post_process_command']
    # note for posterity post_drt_monitor does not like more than 1 proc
    postcommand_list = [mpir_run, mpi_flags, '-np', str(1), post_processor_exec,
                        post_process_command, monitor_file]

    postcommand_string = ' '.join(postcommand_list)
    return postcommand_string

def run(command_string):
    """ Execute passed command

        Args:
            command_string (str): Command to execute
    """
    p = subprocess.Popen(command_string,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         shell=True,
                         universal_newlines=True)

    stdout, stderr = p.communicate()
    print(stdout)
    print(stderr)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
