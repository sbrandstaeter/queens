import SALib
import numpy as np
import math
import random
from pqueens.designers.morris_campolongo_designer import MorrisCampolongoDesigner
from pqueens.sensitivity_analyzers.morris_analyzer import MorrisAnalyzer
from pqueens.example_simulator_functions.borehole_lofi import borehole_lofi
from pqueens.example_simulator_functions.borehole_hifi import borehole_hifi
# to check with SALib
from SALib.sample import morris as mrrs
from SALib.analyze import morris
from SALib.sample import saltelli

# test of my implementation of the Morris method
paramsBorehole =   {   "x1" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 0.05,
                    "max"  : 0.15,
                    "distribution" : 'normal',
                    "distribution_parameter" : [0.05,0.15]
                    },
                    "x2" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 100,
                    "max"  : 50000,
                    "distribution" : 'lognormal',
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

# random seed
seed = 2

MCD = MorrisCampolongoDesigner(paramsBorehole, num_traj ,optim, num_traj_chosen, grid_jump, num_levels,seed)
B_star, perm = MCD.get_all_samples()
Y = np.ones((len(paramsBorehole)+1,num_traj_chosen))
for i in range(num_traj_chosen):
    for j in range(len(paramsBorehole)+1):
        Y[j,i] = borehole_lofi(B_star[i,j,0],B_star[i,j,1],B_star[i,j,2],B_star[i,j,3],B_star[i,j,4],B_star[i,j,5],B_star[i,j,6],B_star[i,j,7])
MA = MorrisAnalyzer(paramsBorehole,num_traj_chosen,grid_jump,num_levels,confidence_level,num_bootstrap_conf)
Si = MA.analyze(B_star, Y, perm)
print('The results with my implementation are :')
S = MA.print_results(Si)

problemBorehole= {
'num_vars': 8,
'names': ['x1', 'x2','x3','x4','x5','x6','x7','x8'],
'bounds' : [[0.05,0.15],[100,50000],[63070,115600],[990,1110],[63.1,116],[700,820],[1120,1680],[9855,12045]],
'function': borehole_lofi,
'groups': None
}
X_SALib = mrrs.sample(problemBorehole, num_traj, num_levels=4, grid_jump=2, optimal_trajectories = 4, local_optimization = False)
Y_SALib =  borehole_lofi(X_SALib[:,0],X_SALib[:,1],X_SALib[:,2],X_SALib[:,3],X_SALib[:,4],X_SALib[:,5],X_SALib[:,6],X_SALib[:,7])
print('The results with SALib are :')
Si_SALib = morris.analyze(problemBorehole, X_SALib, Y_SALib, conf_level=0.95,  print_to_console=True, num_levels=4, grid_jump=2)
