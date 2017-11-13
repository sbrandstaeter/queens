import SALib
import numpy as np
import math
import random
from pqueens.designers.morris_campolongo_designer import MorrisCampolongoDesigner
from pqueens.sensitivity_analyzers.morris_analyzer import MorrisAnalyzer
from pqueens.example_simulator_functions.ishigami import ishigami
# check with the SALib
from SALib.sample import morris as mrrs
from SALib.test_functions import Ishigami
from SALib.analyze import morris
from SALib.sample import saltelli

# test of my implementation of the Morris method
paramsIshigami =   {   "x1" : {
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
# Number of trajectories
num_traj = 1000
# Number of trajectories chosen
num_traj_chosen = 1000
# Grid jump size
grid_jump = 2
# Number of grid level
num_levels = 4
# Value of the confidence level
confidence_level = 0.95
# Do we perform brute-force optimization, if False, num_traj and num_traj_chosen
# must be identical
optim = False
# Number of bootstrap samples to compute the confidence intervals
num_bootstrap_conf = 1000

# random seed
seed = 2

MCD = MorrisCampolongoDesigner(paramsIshigami, num_traj ,optim, num_traj_chosen, grid_jump, num_levels, seed)
B_star, perm = MCD.get_all_samples()
Y = np.ones((len(paramsIshigami)+1,num_traj_chosen))
for i in range(num_traj_chosen):
    for j in range(len(paramsIshigami)+1):
        Y[j,i] = ishigami(B_star[i,j,0],B_star[i,j,1],B_star[i,j,2])
MA = MorrisAnalyzer(paramsIshigami,num_traj_chosen,grid_jump,num_levels,confidence_level,num_bootstrap_conf)
Si = MA.analyze(B_star, Y, perm)
print('The results with my implementation are :')
S = MA.print_results(Si)

# Comparison with the SALib
problemIshigami = {
'num_vars': 3,
'names': ['x1', 'x2','x3'],
'bounds': [[-math.pi, math.pi],[-math.pi, math.pi],[-math.pi, math.pi]],
'function': ishigami,
'groups': None
}
X_SALib = mrrs.sample(problemIshigami, num_traj, num_levels=4, grid_jump=2, optimal_trajectories = None, local_optimization = False)
Y_SALib = Ishigami.evaluate(X_SALib)
print('The results with SALib are :')
Si_SALib = morris.analyze(problemIshigami, X_SALib, Y_SALib, conf_level=0.95,  print_to_console=True, num_levels=4, grid_jump=2)
