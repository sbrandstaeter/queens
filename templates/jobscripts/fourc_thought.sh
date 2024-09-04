#!/bin/bash
source /etc/profile
module load mpi/openmpi-4.1.5
##########################################
#                                        #
#  Specify your paths                    #
#                                        #
##########################################
JOB_ID={{ job_id }}
EXE={{ executable }}
INPUT={{ input_file }}
OUTPUTDIR={{ output_dir }}
OUTPUTPREFIX={{ post_file_prefix }}

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
DoPostprocess=$[ ! -z "{{ post_processor or '' }}" ]
# Note: supported post processor is the  #
#       post_processor.                  #
POSTEXE={{ post_processor }}             #
# Specify everything you need here,      #
# besides the '--file=' as this is       #
# already done by default since it is    #
# clear where your data is stored and    #
# what OUTPUTPREFIX it has!              #
# For detailed information on what can   #
# be specified please use --help         #
POSTOPTIONS={{ post_options or '' }}     #
##########################################


#################################################################
# BEGIN ############### DO NOT TOUCH ME #########################
#################################################################
# This is not a suggestion, this is a rule.
# Talk to admin before touching this section.
source {{ cluster_script or '/lnm/share/donottouch.sh' }}
trap 'EarlyTermination; StageOut' 2 9 15 18
DoChecks
StageIn
RunProgram
wait
RunPostprocessor
wait
StageOut
Show
# END ################## DO NOT TOUCH ME #########################
echo
echo "Main run finished with exit code $PROGRAMEXITCODE"
echo "Post processor finished with exit code $POSTPROCESSOREXITCODE"
echo "Combined exit code is $(CombineExitCodes)"
echo "Job finished at `date`"
exit $(CombineExitCodes)
