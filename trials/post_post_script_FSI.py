#!/home/nitzler/programs/anaconda/anaconda3/envs/py36/bin/python
# coding: utf8

##############################################################################
# This is a post-post process script to extract the QoI from a BACI 
# postprocessed *.mon file using post_drt_monitor
##############################################################################
#import subprocess
import numpy as np
# path to output data is in this example:
# `/home/nitzler/workspace/BACI_Rep/output/first_trial_fsi/*/output/FSI_1`
# already right path for every simulation run!!
def run(path_to_output_data):
    """ Get transient output of QoI """

    # assemble full filename
    my_file = path_to_output_data+'.mon'
    entries = np.loadtxt(my_file, usecols=(1,2), skiprows=4)

    # Discuss with Jonas B. how we want to deal with transient QoI behavior
    # for test purpose just take the maximum for now
    entry_out = entries[-1,1] # this is just the last entry in the dx column

    ####### ERROR DEFINITION #############################
    MAXTIME = 20
    tol = 3
    error = False
    if np.abs(MAXTIME - float(entries[-1,0])) > tol :
        error = True

    return entry_out, error
