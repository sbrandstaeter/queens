import numpy as np
import math
from pqueens.sensitivity_analyzers.sobol_analyzer import SobolAnalyzer
from pqueens.designers.sobol_gratiet_designer import SobolGratietDesigner
from pqueens.example_simulator_functions import sobol_G
import matplotlib.pyplot as plt
# import SALib to verify my own implementation
from SALib.sample import saltelli
from SALib.test_functions import Sobol_G
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

''' Test of the Sobol function'''
paramsSobol =   {   "x1" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 0,
                    "max"  : 1,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [0,1]
                    },
                    "x2" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 0,
                    "max"  : 1,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [0,1]
                    },
                    "x3" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 0,
                    "max"  : 1,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [0,1]
                    },
                    "x4" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 0,
                    "max"  : 1,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [0,1]
                    },
                    "x5" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 0,
                    "max"  : 1,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [0,1]
                    },
                    "x6" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 0,
                    "max"  : 1,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [0,1]
                    },
                    "x7" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 0,
                    "max"  : 1,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [0,1]
                    },
                    "x8" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 0,
                    "max"  : 1,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [0,1]
                    }
                }

# number of sensitivity indices
nb_indices = 2**len(paramsSobol) - 1
Y = np.zeros((output_samples, num_samples, nb_indices+1))
PSD = SobolGratietDesigner(paramsSobol,seed,num_samples)
X = PSD.get_all_samples()
# in case we have several realizations of our gaussian processes
for h in range(output_samples):
    for i in range(nb_indices+1):
        for j in range(num_samples):
            Y[h,j,i] = sobol_G.evaluate(X[j,i,:])
SA = SobolAnalyzer(paramsSobol,calc_second_order, num_bootstrap_samples,
                confidence_level, output_samples)
S = SA.analyze(Y)
print('The results with my implementation are :')
S_print = SA.print_results(S)

# SALib verification
problemSobol = {
'num_vars': 8,
'names': ['x1', 'x2','x3','x4', 'x5','x6','x7', 'x8'],
'bounds' : [[0,1]]*8
}
X_SALib = saltelli.sample(problemSobol, 1000)
Y_SALib = Sobol_G.evaluate(X_SALib)
print('The results with SALib are :')
Si = sobol.analyze(problemSobol, Y_SALib, print_to_console=True)
