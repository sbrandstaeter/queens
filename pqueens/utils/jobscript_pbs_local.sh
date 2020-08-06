#!/bin/bash
##########################################
#                                        #
#  Specify your PBS directives           #
#                                        #
##########################################
# Job name:
#PBS -N {job_name}
# Number of nodes and processors per node (ppn)
#PBS -l nodes=1:ppn={ntasks}
# Walltime: (hours:minutes:seconds)
#PBS -l walltime=6:00:00
# Executing queue
#PBS -q batch
###########################################

# ------------- RUN MAIN JOB IN SINGULARITY (STANDARD JOB SCRIPT) -------------
##########################################
#                                        #
#  Specify your paths                    #
#                                        #
##########################################
WORKDIR=/scratch/PBS_$PBS_JOBID
DESTDIR={DESTDIR}  # Output directory for simulation
EXE={EXE}
INPUT={INPUT}
OUTPUTPREFIX={OUTPUTPREFIX}
##########################################
#                                        #
#       RESTART SPECIFICATION            #
RESTART=0                                #
RESTART_FROM_PREFIX=xxx                  #
##########################################

##########################################
#                                        #
#     POSTPROCESSING SPECIFICATION       #
#                                        #
##########################################
DoPostprocess={POSTPROCESSFLAG}
POSTEXE={POSTEXE}

#################################################################
# BEGIN ############### DO NOT TOUCH ME #########################
#################################################################
# This is not a suggestion, this is a rule.
# Talk to admin before touching this section.
source {CLUSTERSCRIPT}
trap 'EarlyTermination; StageOut' 2 9 15 18
DoChecks
StageIn
RunProgram
wait
#RunPostprocessor # This is still specific to BACI and does not include drt_monitor
# QUEENS Post-processing within jobscript only for post processor without additional options so far
if [ $DoPostprocess = true ]
then
  if [ $RESTART -le 0 ]
  then
    $MPI_RUN $MPIFLAGS -np {nposttasks} $POSTEXE --file=$WORKDIR/$OUTPUTPREFIX
  else
    echo Attention! You are postprocessing files from a restarted simulation. Only the new data is postprocessed, as only this data is available.
    echo
    $MPI_RUN $MPIFLAGS -np {nposttasks} $POSTEXE --file=$WORKDIR/$OUTPUTPREFIX
  fi
fi
wait
StageOut
#Show
echo
echo "Job finished with exit code $? at: `date`"
# END ################## DO NOT TOUCH ME #########################
