#!/bin/sh -f

# Setup shell environment
echo $HOME
cd $HOME
source /etc/profile.d/modules.sh
source $HOME/queens_cluster_suite/load_queens_baci_environment.sh

########################
# GENERAL SPECIFICATIONS
########################
BUILD_DIR=""
RUN_BACI="OFF"
RUN_ENSIGHT_FILTER="OFF"
RUN_QUEENS="ON"

EXE='{{ EXE }}'

#####################
# INPUT SPECIFICATION
#####################
INPUT='{{ INPUT }}'

######################
# OUTPUT SPECIFICATION
######################
OUTPUT_PREFIX={{ OUTPUTPREFIX }}
BACI_OUTPUT_DIR={{ DESTDIR }}
WORKDIR=$BACI_OUTPUT_DIR
ENSIGHT_OUTPUT_DIR=""
ENSIGHT_OPTIONS=""
#######################
# RESTART SPECIFICATION
#######################
RESTART=0
RESTART_FROM_STEP=0                 # <= specify your restart step
RESTART_FROM_DIR=""
RESTART_FROM_PREFIX="" # <= specify the result prefix from which restart is to be read

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
