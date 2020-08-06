#!/bin/bash
##########################################
#                                        #
#  Specify your SLURM directives         #
#                                        #
##########################################
# User's Mail:
#SBATCH --mail-user=name@lnm.mw.tum.de
# When to send mail:
#SBATCH --mail-type=BEGIN,END,FAIL
# Job name:
#SBATCH -J {job_name}
# Standard case: specify only number of cpus
#SBATCH --ntasks={ntasks}
# Walltime: (days-hours:minutes:seconds)
#SBATCH --time=2-00:00:00
# Exclusivity:
#SBATCH --exclusive
# Exclude nodes: (e.g. exclude node07)
# #SBATCH --exclude=node07
###########################################

##########################################
#                                        #
#  Specify your paths                    #
#                                        #
##########################################
WORKDIR=/scratch/SLURM_$SLURM_JOB_ID
DESTDIR={DESTDIR}  # Output directory for simulation
EXE={EXE}
INPUT={INPUT}
OUTPUTPREFIX={OUTPUTPREFIX}
##########################################
#                                        #
#       RESTART SPECIFICATION            #
RESTART=0                                #
RESTART_FROM_PREFIX="test"               #
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
