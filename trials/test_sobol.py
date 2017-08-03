import numpy as np
import math
import os
from combi import *
import matplotlib.pyplot as plt
from pyDOE import *
from random import *
from pqueens.sensitivity_analyzers.sobol_analyzer import SobolAnalyzer
from pqueens.designers.pseudo_saltelli_designer import PseudoSaltelliDesigner
import pqueens.example_simulator_functions.ishigami  as ishigami
import pqueens.example_simulator_functions.currin88_hifi  as currin88_hifi
import pqueens.example_simulator_functions.currin88_lofi  as currin88_lofi
from SALib.analyze import sobol
from SALib.sample import saltelli
from SALib.test_functions import Ishigami
# just a file to test if my two classes work well

dim = 3
num_samples = 1000
num_bootstrap_samples = 100
output_samples = 1
calc_second_order = True
seed = 42
nb_combi = dim+2+math.factorial(dim)//(2*math.factorial(dim-2))

calc_second_order = True
confidence_level = 0.95
num_bootstrap_conf = 100

Y = np.zeros((output_samples, num_samples,nb_combi), dtype = float)

for h in range(output_samples):
    params = { 'X' :{'x1' : np.random.uniform(-math.pi, math.pi,num_samples), 'x2' : np.random.uniform(-math.pi, math.pi,num_samples),'x3' : np.random.uniform(-math.pi, math.pi,num_samples)},
    'X_tilde' :{'x1' : np.random.uniform(-math.pi, math.pi,num_samples), 'x2' : np.random.uniform(-math.pi, math.pi,num_samples),'x3' : np.random.uniform(-math.pi, math.pi,num_samples)}}
    PSD = PseudoSaltelliDesigner(params,seed,num_samples)
    X = PSD.get_all_samples()
    for s in range(nb_combi):
        Y[h,:,s] = ishigami.ishigami(X[s,:,0],X[s,:,1],X[s,:,2])


'''Y = np.zeros(( num_samples,nb_combi), dtype = float)
params = { 'X' :{'x1' : np.random.uniform(-math.pi, math.pi,num_samples), 'x2' : np.random.uniform(-math.pi, math.pi,num_samples),'x3' : np.random.uniform(-math.pi, math.pi,num_samples)},
    'X_tilde' :{'x1' : np.random.uniform(-math.pi, math.pi,num_samples), 'x2' : np.random.uniform(-math.pi, math.pi,num_samples),'x3' : np.random.uniform(-math.pi, math.pi,num_samples)}}
PSD = PseudoSaltelliDesigner(params,seed,num_samples)
X = PSD.get_all_samples()
for s in range(nb_combi):
    Y[:,s] = ishigami.ishigami(X[s,:,0],X[s,:,1],X[s,:,2])'''


SA = SobolAnalyzer(params,calc_second_order, num_bootstrap_samples,
                confidence_level, output_samples,num_bootstrap_conf)
S = SA.analyze(Y)
print(S)

problem = {
'num_vars': 3,
'names': ['x1', 'x2','x3'],
'bounds': [[-math.pi, math.pi]]*3
}

X = saltelli.sample(problem, 1000)
Y = Ishigami.evaluate(X)
Si = sobol.analyze(problem, Y, print_to_console=True)
