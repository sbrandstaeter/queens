#!/usr/bin/env python
# coding: utf8
#PBS -M biehler@lnm.mw.tum.de
#PBS -m abe
#PBS -N queens_run_1
#PBS -l nodes=1:ppn=16
#PBS -l walltime=300:00:00
#PBS -q opteron

################################################################################
#
#  Very basic lauchner script to launch BACI jobs on Kaiser cluster
#  Attention a lot of things are hard coded here that probably should not be
#  hard coded, so proceed with caution.
#                                        #
################################################################################

import os
import subprocess

# get PBS working directory
srcdir=os.environ["PBS_O_WORKDIR"]
os.chdir(srcdir)

BACIDIR=os.environ["HOME"]+'/baci/release'
DESTDIR='/home/biehler/queens_testing/my_first_queens_jobqueens_job_1/output/'


# GENERAL SPECIFICATIONS
PREFIX='queens_run_1'
EXE=BACIDIR + '/baci-release'
EXEP=BACIDIR + '/post_drt_monitor'
INPUT='/home/biehler/input/input2.dat'

# setup MPI environment
MPI_RUN = '/opt/openmpi/1.6.2/gcc48/bin/mpirun'
MPI_HOME = '/opt/openmpi/1.6.2/gcc48'

os.environ["MPI_HOME"] = MPI_HOME
os.environ["MPI_RUN"] = MPI_RUN

# try to create destdir, assume that its there if makedirs fails
try:
    os.makedirs(DESTDIR)
except OSError:
    pass

# determine number of processors from nodefile
#PROCS=16#os.system('cat $PBS_NODEFILE | wc -l')
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
# eventually this should changed to mereyl append the MÜI_HOME path
os.environ["LD_LIBRARY_PATH"] = MPI_HOME


# determine 'optimal' flags for the problem size
if PROCS%16 == 0:
    MPIFLAGS="--mca btl openib,sm,self --mca mpi_paffinity_alone 1"
else:
    MPIFLAGS="--mca btl openib,sm,self"


# Add executable name to the command
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
