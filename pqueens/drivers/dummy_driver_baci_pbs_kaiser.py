#!/home/biehler/miniconda3/bin/python
# coding: utf8

################################################################################
#
#  Very basic lauchner script to launch BACI jobs on Kaiser cluster
#  Attention a lot of things are hard coded here that probably should not be
#  hard coded, so proceed with caution. #!/usr/bin/env python
#                                        #
################################################################################

import os
import subprocess
import json
import sys
import time
from pqueens.database.mongodb import MongoDB
from pqueens.utils.injector import inject
import numpy as np

# all necessary information is passed via this dictionary
driver_options=json.loads(sys.argv[1])

# get PBS working directory
srcdir=os.environ["PBS_O_WORKDIR"]
os.chdir(srcdir)

DESTDIR = str(driver_options['experiment_dir']) + '/' + \
          str(driver_options['job_id'])

PREFIX = str(driver_options['experiment_name']) + '_' + \
         str(driver_options['job_id'])
EXE = driver_options['executable']
EXEP = driver_options['post_processor']

post_process_command = driver_options['post_process_command']

# connect to database and get job parameters
db  = MongoDB(database_address=driver_options['database_address'])
job = db.load(driver_options['experiment_name'], 'jobs',
              {'id' : driver_options['job_id']})

start_time        = time.time()
job['start time'] = start_time

db.save(job, driver_options['experiment_name'], 'jobs',
        {'id' : driver_options['job_id']})

sys.stderr.write("Job launching after %0.2f seconds in submission.\n"
                 % (start_time-job['submit time']))

params = job['params']

output_directory = os.path.join(DESTDIR, 'output')
if not os.path.isdir(output_directory):
    # make complete directory tree
    os.makedirs(output_directory)

# create input file using injector
baci_input_file = DESTDIR + '/' + str(driver_options['experiment_name']) + \
                  '_' + str(driver_options['job_id']) + '.dat'

# create ouput file name
baci_output = output_directory + '/' + str(driver_options['experiment_name']) + \
                  '_' + str(driver_options['job_id'])

# create actual input file in experiment dir folder
inject(params,driver_options['input_template'],baci_input_file)
INPUT = baci_input_file

success = False

# setup MPI environment
MPI_RUN = '/opt/openmpi/1.6.2/gcc48/bin/mpirun'
MPI_HOME = '/opt/openmpi/1.6.2/gcc48'

os.environ["MPI_HOME"] = MPI_HOME
os.environ["MPI_RUN"] = MPI_RUN

# determine number of processors from nodefile
PBS_NODEFILE=os.environ["PBS_NODEFILE"]
print(PBS_NODEFILE)
command_list = ['cat',PBS_NODEFILE,'|','wc', '-l' ]
command_string  = ' '.join(command_list)
p = subprocess.Popen(command_string,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        shell=True,
                        universal_newlines = True)
PROCS, _ = p.communicate()
PROCS = int(PROCS)

# Add non-standard shared library paths
# "LD_LIBRARY_PATH" seems to be also empty, so simply set it to MPI_HOME
# eventually this should changed to mereyl append the MPI_HOME path
os.environ["LD_LIBRARY_PATH"] = MPI_HOME

# determine 'optimal' flags for the problem size
if PROCS%16 == 0:
    MPIFLAGS="--mca btl openib,sm,self --mca mpi_paffinity_alone 1"
else:
    MPIFLAGS="--mca btl openib,sm,self"

# note that we directly write the output to the home folder and do not create
# the appropriate directories on the nodes. This should be changed at some point.
# So long be careful !

runcommand_list = [MPI_RUN , MPIFLAGS, '-np', str(PROCS), EXE, INPUT, baci_output]

runcommand_string = ' '.join(runcommand_list)
#print(runcommand_string)
p = subprocess.Popen(runcommand_string,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        shell=True,
                        universal_newlines = True)

stdout, stderr = p.communicate()
print(stdout)
print(stderr)

monitor_file = '--file=' + str(baci_output)
# note for posterity post_drt_monitor does not like more than 1 proc
postcommand_list = [MPI_RUN , MPIFLAGS, '-np', str(1), EXEP,
                    post_process_command, monitor_file]

postcommand_string = ' '.join(postcommand_list)
#print(postcommand_string)

p = subprocess.Popen(postcommand_string,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        shell=True,
                        universal_newlines = True)

stdout, stderr = p.communicate()
print(stdout)
print(stderr)

line = np.loadtxt(baci_output+'.mon', comments="#",skiprows=4, unpack=False)
# for now simply compute norm of displacement
result = np.sqrt(line[1]**2+line[2]**2+line[3]**2)
print('And the results is: {}'.format(result))

end_time = time.time()

job['result']   = result
job['status']   = 'complete'
job['end time'] = end_time

db.save(job, driver_options['experiment_name'], 'jobs',
        {'id' : driver_options['job_id']})

print('Written result to database')
