##############################################################################
# This is a post-post process script to extract the QoI from a BACI 
# postprocessed *.mon file using post_drt_monitor
##############################################################################
import numpy as np

def run(path_to_output_data):
    """ Get transient output of QoI """

    entry_out = [] # initialize empty list
    for path in [path_to_output_data]:
         entries = np.loadtxt(path+'_1.mon', usecols=(1,3), skiprows=4)
         entry_out=np.append(entry_out, entries[-1,1]) # this is just the last entry in the dx column

        ####### ERROR DEFINITION #############################
         MAXTIME = 9999
         tol = 3
         error = False
         if np.abs(MAXTIME - float(entries[-1,0])) > tol:
                error = False#True
    return entry_out

