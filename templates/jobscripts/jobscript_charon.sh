#!/bin/sh -f
#
###############################
# Specify your SLURM directives
###############################
# Job name:
#SBATCH --job-name {job_name}
#SBATCH --output=slurm-%j-%x.out
#SBATCH --error=slurm-%j-%x.err
#
# Standard case: specify only number of cpus
#SBATCH --ntasks={slurm_ntasks}
#
# Walltime:
#SBATCH --time={walltime}
#
# If you want to specify a certain number of nodes
## and exactly 'ntasks-per-node' cpus on each node.
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=24
#
# For hybrid mpi: e.g. 2 mpi processes each with
# 4 openmp threads
# #SBATCH --ntasks=2
# #SBATCH --cpus-per-task=4
#
# Exclusivity:
#{slurm_exclusive}SBATCH --exclusive
# Exclude nodes: (e.g. exclude node07)
#{slurm_exclude}SBATCH --exclude={slurm_excl_node}
#
# Request specific hardware features
# #SBATCH --constraint="skylake|cascadelake"
###########################################

# Setup shell environment
echo $HOME
cd $HOME
source /etc/profile.d/modules.sh
source $HOME/queens_cluster_suite/load_queens_baci_environment.sh

############################
# SINGULARITY SPECIFICATIONS
############################
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

########################
# GENERAL SPECIFICATIONS
########################
BUILD_DIR=""
RUN_BACI="OFF"
RUN_ENSIGHT_FILTER="OFF"
RUN_QUEENS="ON"

EXE='{EXE}'

#####################
# INPUT SPECIFICATION
#####################
INPUT='{INPUT}'

######################
# OUTPUT SPECIFICATION
######################
OUTPUT_PREFIX={OUTPUTPREFIX}
ENSIGHT_OUTPUT_DIR=""
ENSIGHT_OPTIONS=""
#######################
# RESTART SPECIFICATION
#######################
RESTART=0
RESTART_FROM_STEP=0                 # <= specify your restart step
RESTART_FROM_DIR=""
RESTART_FROM_PREFIX="" # <= specify the result prefix from which restart is to be read

###############################
# POST PROCESSING SPECIFICATION
###############################
DoDataProcessing={DATAPROCESSINGFLAG} # post- and data-processing flag for singularity run

#################################################################
# BEGIN ############### DO NOT TOUCH ME #########################
#################################################################
# execute program
source $HOME/queens_cluster_suite/queens_job_core
trap 'early; stageout' 2 9 15 18
dochecks
stagein
runprogram
wait
stageout
show
# END ################## DO NOT TOUCH ME #########################
echo
echo "Job finished with exit code $? at: `date`"
# ------- FINISH AND CLEAN SINGULARITY JOB (DONE ON MASTER/LOGIN NODE!) -------
wait
# Post- and data-processing for singularity run
# (cd back into home since pwd does not exist anymore)
if [ $DoDataProcessing = true ]
then
  $MPI_RUN $MPIFLAGS -np {nposttasks} $EXE $INPUT '--post=true'
fi
# END ################## DO NOT TOUCH ME #########################
