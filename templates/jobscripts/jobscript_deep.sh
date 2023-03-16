#!/bin/bash
##########################################
#                                        #
#  Specify your PBS directives           #
#                                        #
##########################################
# Job name:
#PBS -N {{ job_name }}
# Number of nodes and processors per node (ppn)
#PBS -l nodes={{ pbs_nodes }}:ppn={{ pbs_ppn }}
# Walltime: (hours:minutes:seconds)
#PBS -l walltime={{ walltime }}
# Executing queue
#PBS -q {{ pbs_queue }}
###########################################

##########################################
#                                        #
#  Specify your paths                    #
#                                        #
##########################################
WORKDIR=/scratch/PBS_$PBS_JOBID
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
