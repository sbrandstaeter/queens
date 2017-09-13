import SALib
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from pqueens.designers.morris_campolongo_designer import MorrisCampolongoDesigner
from pqueens.sensitivity_analyzers.morris_analyzer import MorrisAnalyzer
from pqueens.example_simulator_functions import sobol_G
# check the implementation with SALib
from SALib.test_functions import Sobol_G
from SALib.sample import morris as mrrs
from SALib.analyze import morris
from SALib.sample import saltelli

# test of my implementation of the Morris method
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
                    }}

# Number of trajectories
num_traj = 10
# Number of trajectories chosen
num_traj_chosen = 4
# Grid jump size
grid_jump = 2
# Number of grid level
num_levels = 4
# Value of the confidence level
confidence_level = 0.95
# Do we perform brute-force optimization, if False, num_traj and num_traj_chosen
# must be identical
optim = True
# Number of bootstrap samples to compute the confidence intervals
num_bootstrap_conf = 1000

MCD = MorrisCampolongoDesigner(paramsSobol, num_traj ,optim, num_traj_chosen, grid_jump, num_levels)
B_star, perm = MCD.get_all_samples()
Y = np.ones((len(paramsSobol)+1,num_traj_chosen))
for i in range(num_traj_chosen):
    for j in range(len(paramsSobol)+1):
        Y[j,i]  = sobol_G.evaluate(B_star[i,j,:])
MA = MorrisAnalyzer(paramsSobol,num_traj_chosen,grid_jump,num_levels,confidence_level,num_bootstrap_conf)
Si = MA.analyze(B_star, Y, perm)
print('The results with my implementation are :')
S = MA.print_results(Si)

# Comparison with the SALib
problemSobol = {
'num_vars': 8,
'names': ['x1', 'x2','x3','x4', 'x5','x6','x7','x8'],
'bounds' : [[0,1]]*8,
'groups': None
}
X_SALib = mrrs.sample(problemSobol, num_traj, num_levels=4, grid_jump=2, optimal_trajectories = 4, local_optimization = False)
Y_SALib = Sobol_G.evaluate(X_SALib)
print('The results with SALib are :')
Si_SALib = morris.analyze(problemSobol, X_SALib, Y_SALib, conf_level=0.95,  print_to_console=True, num_levels=4, grid_jump=2)
