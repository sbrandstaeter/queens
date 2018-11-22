#!/home/nitzler/programs/anaconda/anaconda3/envs/py36/bin/python
# coding: utf8


################################################################################
#
#  Very basic lauchner script to launch BACI jobs on Bruteforce cluster
#  Attention a lot of things are hard coded here that probably should not be
#  hard coded, so proceed with caution. #!/usr/bin/env python
#                                        #
################################################################################
import os
import subprocess
import json
import sys
import time
import importlib.util
from pqueens.database.mongodb import MongoDB
from pqueens.utils.injector import inject

def main(args):
    """
        Very basic lauchner script to launch BACI jobs on Bruteforce cluster.

        Attention a lot of things are hard coded here that probably should not be
        hard coded, so proceed with caution.

        Args:
            args (JSON document): file-like object containing a JSON document
    """
    # The following is necessary to fix JSON FORMAT reader
    args=args.replace('\\', '\"')
    # all necessary information is passed via this dictionary
    driver_options = json.loads(args)

    # get SLURM working directory
    srcdir = os.environ["SLURM_SUBMIT_DIR"]
    print(srcdir)
    os.chdir(srcdir)
    #print(args)
    # connect to database and get job parameters
    db = MongoDB(database_address='10.10.0.1:27017')
    job = init_job(driver_options, db)
    _, baci_input_file, baci_output = setup_dirs_and_files(driver_options)
     #creat actual input file in experiment dir folder
    inject(job['params'], driver_options['input_template'], baci_input_file)

    # assemble command to run BACI
    runcommand_string, my_env = get_runcommand_string(driver_options, baci_input_file, baci_output)
    #run BACI
    #runcommand_string = 'cd $HOME && module purge && module load intel-studio-2016 mpi/openmpi/intel/1.10.1 && LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_HOME/lib && export LD_LIBRARY_PATH && ' + runcommand_string
    run(runcommand_string, my_env)

    # do postprocessing
    do_postprocessing(driver_options, baci_output)

    my_file = open(baci_output, 'r')
    result = my_file.readline()
    finish_job(driver_options, db, job, result)

def get_num_nodes():
    """ determine number of processors from nodefile """
    slurm_nodefile = os.environ["SLURM_JOB_NODELIST"]
    #print(slurm_nodefile)
    command_list = ['cat', slurm_nodefile, '|', 'wc', '-l']
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
    mpi_run = '/cluster/mpi/intel/openmpi/1.10.1/bin/mpirun'
    mpi_home = '/cluster/mpi/intel/openmpi/1.10.1'

    os.environ["MPI_HOME"] = mpi_home
    os.environ["MPI_RUN"] = mpi_run
    os.environ["LD_LIBRARY_PATH"] += "/cluster/mpi/intel/openmpi/1.10.1'/lib"

    # Add non-standard shared library paths
    # "LD_LIBRARY_PATH" seems to be also empty, so simply set it to MPI_HOME
    # eventually this should changed to mereyl append the MPI_HOME path
    my_env = os.environ.copy()
    # determine 'optimal' flags for the problem size
    if num_procs%16 == 0:
        mpi_flags = "--mca btl openib,sm,self --mca mpi_paffinity_alone 1"
    else:
        mpi_flags = "--mca btl openib,sm,self"

    return mpi_run, mpi_flags, my_env

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
        # make complete directory treesr/sbin:/sbin:"
        os.makedirs(output_directory)

    # create input file using injector
    baci_input_file = dest_dir + '/' + str(driver_options['experiment_name']) + \
                      '_' + str(driver_options['job_id']) + '.dat'

    # create ouput file name
    baci_output =  output_directory + '/' + str(driver_options['experiment_name']) + \
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

#def do_postpostprocessing(driver_options, baci_output):
#    """ Execute post post processing step
#
#        Args:
#            driver_options (dict): Options dictionary
#            baci_output (str): Path to BACI output files
#
#        Returns:
#            float: Postprocessed result
#    """
#    post_post_script = driver_options.get('post_post_script', None)
#    result = None
#    if post_post_script != None:
#        spec = importlib.util.spec_from_file_location("module.name", post_post_script)
#    print(runcommand_string)
#        post_post_proc = importlib.util.module_from_spec(spec)
#        spec.loader.exec_module(post_post_proc)
#        result = post_post_proc.run(baci_output)
#        print('Got result: {}'.format(result))
#    else:
#        raise RuntimeError("You need to provide post_post_script in the driver "
#                           "driver_params section of the config file to get results")
#
#    return result
#

def get_runcommand_string(driver_options, baci_input_file, baci_output):
    """ Assemble run command for BACI

        Args:
            driver_options (dict): Options dictionary
            baci_input_file (str): String with baci_input_file
            baci_output     (str): String with output file/path

        Returns:
            str: Complete command to execute BACI
    """
    procs = 4
    mpir_run, mpi_flags, my_env = setup_mpi(procs)
    executable = driver_options['path_to_executable']

    # note that we directly write the output to the home folder and do not create
    # the appropriate directories on the nodes. This should be changed at some point.
    # So long be careful !
    # TODO: Check MPI run below, I commented it out but probably necessary?
   # runcommand_list = [mpir_run, mpi_flags, '-np', str(procs), executable,
   #                       baci_input_file, baci_output]
    runcommand_list = [mpir_run, mpi_flags, executable,
                       baci_input_file, baci_output]

    runcommand_string = ' '.join(runcommand_list)
    return runcommand_string, my_env

def do_postprocessing(driver_options, baci_output):
    """ Assemble post processing command for BACI

        Args:
            driver_options (dict): Options dictionary
            baci_output (str):     Path to BACI output file
    """
    runcommand_string, my_env = get_postcommand_string(driver_options, baci_output)
    #runcommand_string = 'cd $HOME && module purge && module load intel-studio-2016 mpi/openmpi/intel/1.10.1 && LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_HOME/lib && export LD_LIBRARY_PATH && ' + runcommand_string

    if runcommand_string != None:
        run(runcommand_string, my_env)

def get_postcommand_string(driver_options, baci_output):
    """ Assemble post processing command for BACI

        Args:
            driver_options (dict): Options dictionary
            baci_output (str):     Path to BACI output file

        Returns:
            str: Post processing command for BACI
    """
    procs = get_num_nodes()
    mpir_run, mpi_flags,my_env = setup_mpi(procs)
    post_processor_exec = driver_options.get('path_to_postprocessor', None)
    postcommand_string = None
    if post_processor_exec != None:
        monitor_file = '--file=' + str(baci_output)
        post_process_command = driver_options.get('post_process_command', "")
        # note for posterity post_drt_monitor does not like more than 1 proc
        # TODO: CHECK MPI here
        postcommand_list = [mpir_run, mpi_flags, post_processor_exec, post_process_command, monitor_file]
        #postcommand_list = [post_processor_exec,
                           # post_process_command, monitor_file]


        postcommand_string = ' '.join(postcommand_list)

    return postcommand_string, my_env


def run(command_string,my_env):
    """ Execute passed command

        Args:
            command_string (str): Command to execute
    """
    print(command_string)
    p = subprocess.Popen(command_string,
                         env=my_env,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         shell=True,
                         universal_newlines=True)

    stdout, stderr = p.communicate()
    print(stderr)
    print(stdout)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
