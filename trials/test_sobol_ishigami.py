import numpy as np
import math
from pqueens.sensitivity_analyzers.sobol_analyzer import SobolAnalyzer
from pqueens.designers.sobol_gratiet_designer import SobolGratietDesigner
from pqueens.example_simulator_functions.ishigami import ishigami
import matplotlib.pyplot as plt
# import SALib to verify my own implementation
from SALib.sample import saltelli
from SALib.test_functions import Ishigami
from SALib.analyze import sobol
# set up all necessary parameters for SA
# dimension of input space
num_samples = 1000
num_bootstrap_samples = 100
# number of realization of emulator
output_samples = 1
# do we want to compute second order indices
calc_second_order = True
# fix seed for random number generation
seed = 2

# value of confidence_level, should be between 0 and 1
confidence_level = 0.95


''' Test of the Ishigami function'''
params =   {   "x1" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : -math.pi,
                    "max"  : math.pi,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [-math.pi,math.pi]
                    },
                    "x2" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : -math.pi,
                    "max"  : math.pi,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [-math.pi,math.pi]
                    },
                    "x3" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : -math.pi,
                    "max"  : math.pi,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [-math.pi,math.pi]},
                }

# number of sensitivity indicesfrom SALib.analyze import sobol
nb_indices = 2**len(params) - 1
Y = np.zeros((output_samples, num_samples, nb_indices+1))
PSD = SobolGratietDesigner(params,seed,num_samples)
X = PSD.get_all_samples()

# in case we have several realizations of our gaussian processes
for h in range(output_samples):
    for s in range(nb_indices):
        Y[h,:,s] = ishigami(X[:,s,0],X[:,s,1],X[:,s,2])
SA = SobolAnalyzer(params,calc_second_order, num_bootstrap_samples,
                confidence_level, output_samples)
S = SA.analyze(Y)
print('The results with my implementation are :')
S_print = SA.print_results(S)

# verify with SALib
problem = {
'num_vars': 3,
'names': ['x1', 'x2','x3'],
'bounds': [[-math.pi, math.pi]]*3
}
X_SALib = saltelli.sample(problem, num_samples)
X = np.ones((num_samples, nb_indices+1, len(params)))
for i in range(num_samples):
    X[i,:,:] = X_SALib[i*(nb_indices+1):i*(nb_indices+1)+nb_indices+1,:]
Y_SALib = Ishigami.evaluate(X_SALib)
print('The results with SALib are :')
Si_SALib = sobol.analyze(problem, Y_SALib, print_to_console=True)
