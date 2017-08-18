import SALib
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from SALib.sample import morris as mrrs
from SALib.analyze import morris
from SALib.sample import saltelli
from pqueens.designers.morris_campolongo_designer import MorrisCampolongoDesigner
from pqueens.sensitivity_analyzers.morris_analyzer import MorrisAnalyzer
from pqueens.sensitivity_analyzers.SALiblocal.SALib.test_functions import Sobol_G_me
from SALib.test_functions import Sobol_G

# test of my implementation of the Morris function
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
Y = np.ones((len(paramsSobol)+1,num_traj_chosen,))
for i in range(num_traj_chosen):
    for j in range(len(paramsSobol)+1):
        Y[j,i]  = Sobol_G_me.evaluate(B_star[i,j,:])
MA = MorrisAnalyzer(paramsSobol,num_traj_chosen,grid_jump,num_levels,confidence_level,num_bootstrap_conf)
Si = MA.analyze(B_star, Y, perm)
print('The results with my implementation are :')
S = MA.print_results(Si)
plt.plot(Si['mu_star'], Si['sigma'],'ro')
plt.xlabel('mu*')
plt.ylabel('Sigma')
plt.axis([0, 15, 0, 15])
plt.text(Si['mu_star'][0]+0.015, Si['sigma'][0]+0.015,  r'x1')
plt.text(Si['mu_star'][1]+0.015, Si['sigma'][1]+0.015,  r'x2')
plt.text(Si['mu_star'][2]+0.015, Si['sigma'][2]+0.015,  r'x3')
plt.text(Si['mu_star'][3]+0.015, Si['sigma'][3]+0.015,  r'x4')
plt.text(Si['mu_star'][4]+0.015, Si['sigma'][4]+0.015,  r'x5')
plt.text(Si['mu_star'][5]+0.015, Si['sigma'][5]+0.015,  r'x6')
plt.text(Si['mu_star'][6]+0.015, Si['sigma'][6]+0.015,  r'x7')
plt.text(Si['mu_star'][7]+0.015, Si['sigma'][7]+0.015,  r'x8')

plt.title('Morris Sensitivity Indices ')
plt.show()

# Comparison with the SALib

problemSobol = {
'num_vars': 8,
'names': ['x1', 'x2','x3','x4', 'x5','x6','x7', 'x8'],
'bounds' : [[0,1]]*8,
'groups': None
}

X_SALib = mrrs.sample(problemSobol, num_traj, num_levels=4, grid_jump=2, optimal_trajectories = None, local_optimization = False)
Y_SALib = Sobol_G.evaluate(X_SALib)
print('The results with SALib are :')
Si_SALib = morris.analyze(problemSobol, X_SALib, Y_SALib, conf_level=0.95,  print_to_console=True, num_levels=4, grid_jump=2)
