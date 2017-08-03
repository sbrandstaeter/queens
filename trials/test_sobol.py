import numpy as np
import math
from pqueens.sensitivity_analyzers.sobol_analyzer import SobolAnalyzer
from pqueens.designers.pseudo_saltelli_designer import PseudoSaltelliDesigner
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
num_bootstrap_samples = 1
# numeber of realization of emulator
output_samples = 1
# do we want to compute second order indices
calc_second_order = False
# fix seed for random number generation
seed = 42

nb_combi = dim+2+math.factorial(dim)//(2*math.factorial(dim-2))

confidence_level = 0.95
# TODO check for what this is needed ?
num_bootstrap_conf = 1

Y = np.zeros((output_samples, num_samples,nb_combi))

for h in range(output_samples):
    params = { 'X' :{'x1' : np.random.uniform(-math.pi, math.pi,num_samples), 'x2' : np.random.uniform(-math.pi, math.pi,num_samples),'x3' : np.random.uniform(-math.pi, math.pi,num_samples)},
    'X_tilde' :{'x1' : np.random.uniform(-math.pi, math.pi,num_samples), 'x2' : np.random.uniform(-math.pi, math.pi,num_samples),'x3' : np.random.uniform(-math.pi, math.pi,num_samples)}}

    PSD = PseudoSaltelliDesigner(params,seed,num_samples)
    X = PSD.get_all_samples()
    for s in range(nb_combi):
        Y[h,:,s] = ishigami.ishigami(X[s,:,0],X[s,:,1],X[s,:,2])


# test sensitivity analysis in combination with stochastic emulator
# Y = np.zeros(( num_samples,nb_combi), dtype = float)
# params = { 'X' :{'x1' : np.random.uniform(-math.pi, math.pi,num_samples), 'x2' : np.random.uniform(-math.pi, math.pi,num_samples),'x3' : np.random.uniform(-math.pi, math.pi,num_samples)},
#     'X_tilde' :{'x1' : np.random.uniform(-math.pi, math.pi,num_samples), 'x2' : np.random.uniform(-math.pi, math.pi,num_samples),'x3' : np.random.uniform(-math.pi, math.pi,num_samples)}}
# PSD = PseudoSaltelliDesigner(params,seed,num_samples)
# X = PSD.get_all_samples()
# for s in range(nb_combi):
#     Y[:,s] = ishigami.ishigami(X[s,:,0],X[s,:,1],X[s,:,2])


SA = SobolAnalyzer(params,calc_second_order, num_bootstrap_samples,
                confidence_level, output_samples,num_bootstrap_conf)
S = SA.analyze(Y)
print(S)

# verify with SALib

problem = {
'num_vars': 3,
'names': ['x1', 'x2','x3'],
'bounds': [[-math.pi, math.pi]]*3
}

X = saltelli.sample(problem, 1000)
Y = Ishigami.evaluate(X)
Si = sobol.analyze(problem, Y, print_to_console=True)
