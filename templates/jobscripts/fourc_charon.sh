#!/bin/bash
# Setup shell environment and start from home dir
echo $HOME
cd $HOME
source /etc/profile.d/modules.sh
source /home/opt/cluster_tools/core/load_baci_environment.sh

module list
##########################################
#                                        #
#  Specify the paths                     #
#                                        #
##########################################

RUN_BACI="ON"
EXE={{ executable }}
BACI_BUILD_DIR="$(dirname "${EXE}")"

INPUT={{ input_file }}
BACI_OUTPUT_DIR={{ output_dir }}
OUTPUTPREFIX={{ post_file_prefix }}


##########################################
#                                        #
#  Postprocessing                        #
#                                        #
##########################################
DoPostprocess=$[ ! -z "{{ post_processor }}" ]
if [ $DoPostprocess ]
then
  RUN_ENSIGHT_FILTER="ON"
else
  RUN_ENSIGHT_FILTER="OFF"
fi

ENSIGHT_OUTPUT_DIR={{ output_dir }}
ENSIGHT_OPTIONS={{ post_options }}


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
