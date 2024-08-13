#!/bin/bash
module load mpi/openmpi/gcc/4.1.5
##########################################
#                                        #
#  Specify your paths                    #
#                                        #
##########################################
JOB_ID={{ JOB_ID }}
EXE={{ EXE }}
INPUT={{ INPUT }}
OUTPUTDIR={{ DESTDIR }}
OUTPUTPREFIX={{ OUTPUTPREFIX }}

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
DoPostprocess={{ POSTPROCESS }}          #
# Note: supported post processor is the  #
#       post_processor.                  #
POSTEXE={{ POSTEXE }}                    #
# Specify everything you need here,      #
# besides the '--file=' as this is       #
# already done by default since it is    #
# clear where your data is stored and    #
# what OUTPUTPREFIX it has!              #
# For detailed information on what can   #
# be specified please use --help         #
POSTOPTIONS={{ POSTOPTIONS }}            #
##########################################


#################################################################
# BEGIN ############### DO NOT TOUCH ME #########################
#################################################################
# This is not a suggestion, this is a rule.
# Talk to admin before touching this section.
source {{ CLUSTERSCRIPT }}
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
