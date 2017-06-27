#!/usr/bin/env python
# coding: utf8

################################################################################
#
#  Very basic lauchner script to launch BACI jobs on Kaiser cluster
#  Attention a lot of things are hard coded here that probably should not be
#  hard coded, so proceed with caution.
#                                        #
################################################################################

import os
import subprocess
import json
from sys import argv

# all necessary information is passed via this dictionary
driver_options=json.loads(argv[1])
#print(driver_options)

# get PBS working directory
srcdir=os.environ["PBS_O_WORKDIR"]
os.chdir(srcdir)

DESTDIR = driver_options['experiment_dir']
PREFIX = str(driver_options['experiment_name']) + '_' + str(driver_options['job_id'])
EXE = driver_options['executable']
EXEP = driver_options['post_processor']
INPUT=driver_options['input_template']
post_process_command = driver_options['post_process_command']

# TODO connect to database
# TODO get input parameters from database
# TODO create actual input file using provided template

# setup MPI environment
MPI_RUN = '/opt/openmpi/1.6.2/gcc48/bin/mpirun'
MPI_HOME = '/opt/openmpi/1.6.2/gcc48'

os.environ["MPI_HOME"] = MPI_HOME
os.environ["MPI_RUN"] = MPI_RUN

output_directory = os.path.join(DESTDIR, 'output')
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

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
print(PROCS)

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

runcommand_list = [MPI_RUN , MPIFLAGS, '-np', str(PROCS), EXE, INPUT, DESTDIR]

runcommand_string = ' '.join(runcommand_list)
print(runcommand_string)
p = subprocess.Popen(runcommand_string,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        shell=True,
                        universal_newlines = True)

stdout, stderr = p.communicate()
print(stdout)
print(stderr)

# TODO postprocess results
# TODO write results to database
