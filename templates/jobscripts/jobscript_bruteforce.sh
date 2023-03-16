#!/bin/bash
##########################################
#                                        #
#  Specify your SLURM directives         #
#                                        #
##########################################
# Job name:
#SBATCH -J {{ job_name }}
# Standard case: specify only number of cpus
#SBATCH --ntasks={{ slurm_ntasks }}
# Walltime: (hours:minutes:seconds)
#SBATCH --time={{ walltime }}
###########################################

##########################################
#                                        #
#  Specify your paths                    #
#                                        #
##########################################
WORKDIR=/scratch/SLURM_$SLURM_JOB_ID
DESTDIR={{ DESTDIR }}  # output directory for run
EXE='{{ EXE }}' # either CAE executable or singularity image
INPUT='{{ INPUT }}'  # either input file or, for singularity, list of arguments specifying run
OUTPUTPREFIX={{ OUTPUTPREFIX }}
##########################################
#                                        #
#       RESTART SPECIFICATION            #
RESTART=0                                #
RESTART_FROM_PREFIX=xxx                  #
##########################################

##########################################
#                                        #
#     POST- AND DATA-PROCESSING          #
#     SPECIFICATION                      #
#                                        #
##########################################
DoDataProcessing={{ DATAPROCESSINGFLAG }} # post- and data-processing flag for singularity run

#################################################################
# BEGIN ############### DO NOT TOUCH ME #########################
#################################################################
# This is not a suggestion, this is a rule.
# Talk to admin before touching this section.
source {{ CLUSTERSCRIPT }}
trap 'EarlyTermination; StageOut' 2 9 15 18
LoadBACIModules
DoChecks
StageIn
RunProgram
wait
StageOut
#Show
echo
echo "Job finished with exit code $? at: `date`"
# ------- FINISH AND CLEAN SINGULARITY JOB (DONE ON MASTER/LOGIN NODE!) -------
wait
# Post- and data-processing for singularity run
# (cd back into home since pwd does not exist anymore)
if [ $DoDataProcessing = true ]
then
  $MPI_RUN $MPIFLAGS -np {{ nposttasks }} $EXE $INPUT $WORKDIR '--post=true'
fi
# END ################## DO NOT TOUCH ME #########################
