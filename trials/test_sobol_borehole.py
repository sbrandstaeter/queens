import numpy as np
import math
from pqueens.sensitivity_analyzers.sobol_analyzer import SobolAnalyzer
from pqueens.designers.sobol_gratiet_designer import SobolGratietDesigner
from pqueens.example_simulator_functions.borehole_lofi import borehole_lofi
from pqueens.example_simulator_functions.borehole_hifi import borehole_hifi
import matplotlib.pyplot as plt
# import SALib to verify my own implementation
from SALib.sample import saltelli
from SALib.analyze import sobol

# set up all necessary parameters for SA
num_samples = 1000
num_bootstrap_samples = 1000
# number of realization of emulator
output_samples = 1
# do we want to compute second order indices
calc_second_order = True
# fix seed for random number generation
seed = 42
# value of confidence_level, should be between 0 and 1
confidence_level = 0.95

''' Test of the Borehole function'''
paramsBorehole =   {   "x1" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 0.05,
                    "max"  : 0.15,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [0.05,0.15]
                    },
                    "x2" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 100,
                    "max"  : 50000,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [100,50000]
                    },
                    "x3" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 63070,
                    "max"  : 115600,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [63070,115600]
                    },
                    "x4" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 990,
                    "max"  : 1110,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [990,1110]
                    },
                    "x5" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 63.1,
                    "max"  : 116,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [63.1,116]
                    },
                    "x6" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 700,
                    "max"  : 820,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [700,820]
                    },
                    "x7" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 1120,
                    "max"  : 1680,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [1120,1680]
                    },
                    "x8" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 9855,
                    "max"  : 12045,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [9855,12045]
                    }
                }

nb_indices = 2**len(paramsBorehole) - 1
Y = np.zeros((output_samples, num_samples, nb_indices+1))
PSD = SobolGratietDesigner(paramsBorehole,seed,num_samples)
X = PSD.get_all_samples()

# in case we have several realizations of our gaussian processes
for h in range(output_samples):
    for s in range(1+(len(paramsBorehole)*(len(paramsBorehole)+1))//2):
        print("s first{}".format(s))
        #print("my_samples {}".format(X[:,s,0],X[:,s,1],X[:,s,2],X[:,s,3],X[:,s,4],X[:,s,5],X[:,s,6],X[:,s,7]))
        #exit()
        Y[h,:,s] = borehole_lofi(X[:,s,0],X[:,s,1],X[:,s,2],X[:,s,3],X[:,s,4],X[:,s,5],X[:,s,6],X[:,s,7])
    for s in range(nb_indices-len(paramsBorehole)-1,nb_indices+1):
        print("s second {}".format(s))
        Y[h,:,s] = borehole_lofi(X[:,s,0],X[:,s,1],X[:,s,2],X[:,s,3],X[:,s,4],X[:,s,5],X[:,s,6],X[:,s,7])
SA = SobolAnalyzer(paramsBorehole,calc_second_order, num_bootstrap_samples,
                confidence_level, output_samples)
S = SA.analyze(Y)
print('The results with my implementation are :')
S_print = SA.print_results(S)

#Verification with the SALib implementation
problemBorehole = {
'num_vars': 8,
'names': ['x1', 'x2','x3','x4', 'x5','x6','x7', 'x8'],
'bounds' : [[0.05,0.15],[100,50000],[63070,115600],[990,1110],[63.1,116],[700,820],[1120,1680],[9855,12045]]
}
print('The results with SALib are :')
X = saltelli.sample(problemBorehole, 1000)
print("X {}".format(X.shape))
Y = borehole_hifi(X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5],X[:,6],X[:,7])
Si = sobol.analyze(problemBorehole, Y, print_to_console=True)
