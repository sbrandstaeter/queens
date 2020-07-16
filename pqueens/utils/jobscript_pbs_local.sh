 #!/bin/h
##########################################
#                                        #
#  Specify your PBS directives           #
#                                        #
##########################################
#PBS -N {job_name}
#PBS -l nodes=1:ppn={ntasks}
#PBS -l walltime=6:00:00
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
RESTART=0                                # <= specify your restart step
RESTART_FROM_PREFIX=xxx                  # <= specify the result prefix from which restart is to be read
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
