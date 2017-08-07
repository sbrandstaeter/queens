import numpy as np
import math
from pqueens.sensitivity_analyzers.sobol_analyzer import SobolAnalyzer
from pqueens.designers.sobol_gratiet_designer import PseudoSaltelliDesigner
import pqueens.example_simulator_functions.ishigami  as ishigami


# import SALib to verify own implementation
from SALib.analyze import sobol
from SALib.sample import saltelli
from SALib.test_functions import Ishigami


# set up all necessary parameters for SA
# dimension of input space
dim = 3
#
num_samples = 1000
num_bootstrap_samples = 100
# number of realization of emulator
output_samples = 1
# do we want to compute second order indices
calc_second_order = True
# fix seed for random number generation
seed = 42

nb_combi = dim+2+math.factorial(dim)//(2*math.factorial(dim-2))

confidence_level = 0.95

Y = np.zeros((output_samples, num_samples,nb_combi))

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
                    "distribution_parameter" : [-math.pi,math.pi]}
                }

PSD = PseudoSaltelliDesigner(params,seed,num_samples)
X = PSD.get_all_samples()
# in case we have several realizations of our gaussian processes
for h in range(output_samples):
    for s in range(nb_combi):
        Y[h,:,s] = ishigami.ishigami(X[:,s,0],X[:,s,1],X[:,s,2])

SA = SobolAnalyzer(params,calc_second_order, num_bootstrap_samples,
                confidence_level, output_samples)
S = SA.analyze(Y)
S_print = SA.print_results(S)

# verify with SALib

problem = {
'num_vars': 3,
'names': ['x1', 'x2','x3'],
'bounds': [[-math.pi, math.pi]]*3
}

X = saltelli.sample(problem, 1000)
Y = Ishigami.evaluate(X)
# Si = sobol.analyze(problem, Y, print_to_console=True)
