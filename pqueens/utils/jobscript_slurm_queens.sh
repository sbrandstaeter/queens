#!/bin/bash
##########################################
#                                        #
#  Specify your SLURM directives         #
#                                        #
##########################################
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
EXE={EXE}  # triggers mainrun of singularity image
INPUT='{INPUT} --workdir '  # for singularity just the args flags for the image that specify the run
OUTPUTPREFIX=''
##########################################
#                                        #
#       RESTART SPECIFICATION            #
RESTART=0                                #
RESTART_FROM_PREFIX=xxx                  #
##########################################

##########################################
#                                        #
#     POSTPROCESSING SPECIFICATION       #  # NOT NEEDED FOR QUEENS
#                                        #
DoPostprocess=false                      #
# Note: supported post processor is the  #
#       post_processor.                  #
POSTEXE=''                               #
# Specify everything you need here,      #
# besides the '--file=' as this is       #
# already done by default since it is    #
# clear where your data is stored and    #
# what OUTPUTPREFIX it has!              #
# For detailed information on what can   #
# be specified please use --help         #
POSTOPTIONS='--filter="ensight"'         #
##########################################


#################################################################
# BEGIN ############### DO NOT TOUCH ME #########################
#################################################################
# This is not a suggestion, this is a rule.
# Talk to admin before touching this section.
source /lnm/share/donottouch.sh
trap 'EarlyTermination; StageOut' 2 9 15 18
DoChecks
StageIn
RunProgram
wait
RunPostprocessor # This is still specific to BACI and does not include drt_monitor
wait
StageOut
Show
# END ################## DO NOT TOUCH ME #########################
echo
echo "Job finished with exit code $? at: `date`"

# ------- FINISH AND CLEAN SINGULARITY JOB (DONE ON MASTER/LOGIN NODE!) -------
wait
# cd back into home since pwd does not exist anymore
$MPI_RUN $MPIFLAGS -np 1 $EXE $INPUT $WORKDIR '--post=true'
