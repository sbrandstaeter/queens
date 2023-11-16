#!/bin/bash
# Setup shell environment and start from home dir
echo $HOME
cd $HOME
source /etc/profile.d/modules.sh
source /home/opt/cluster_tools/core/load_baci_environment_23_10.sh

module list
##########################################
#                                        #
#  Specify the paths                     #
#                                        #
##########################################

RUN_BACI="ON"
BACI_BUILD_DIR={{ BUILDDIR }}
EXE={{ EXE }}

INPUT={{ INPUT }}
BACI_OUTPUT_DIR={{ DESTDIR }}
OUTPUT_PREFIX={{ OUTPUTPREFIX }}


##########################################
#                                        #
#  Postprocessing                        #
#                                        #
##########################################
DoPostprocess={{ POSTPROCESS }}
if [ $DoPostprocess = true ]
then
  RUN_ENSIGHT_FILTER="ON"
else
  RUN_ENSIGHT_FILTER="OFF"
fi

ENSIGHT_OUTPUT_DIR={{ DESTDIR }}
ENSIGHT_OPTIONS={{ POSTOPTIONS }}


##########################################
#                                        #
#  RESTART SPECIFICATION                 #
#                                        #
##########################################

RESTART_FROM_STEP=0                 	# specify the restart step here and in .datfile
RESTART_FROM_DIR=""			# same as output
RESTART_FROM_PREFIX="" 		# prefix typically s

#################################################################
# BEGIN ############### DO NOT TOUCH ME #########################
#################################################################

# execute program
source /home/opt/cluster_tools/core/charon_job_core
trap 'EarlyTermination; StageOut' 2 9 15 18
DoChecks
StageIn
RunProgram
wait
StageOut
# show
# END ################## DO NOT TOUCH ME #########################
echo
echo "Job finished with exit code $? at: `date`"
