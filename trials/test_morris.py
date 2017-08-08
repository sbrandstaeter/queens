import SALib
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from SALib.sample import morris as mrrs
from SALib.test_functions import Ishigami
from SALib.analyze import morris
from SALib.analyze import sobol
from SALib.sample import saltelli
from SALib.test_functions import Sobol_G
from pqueens.designers.morris_campolongo_designer import MorrisCampolongoDesigner
from pqueens.sensitivity_analyzers.morris_analyzer import MorrisAnalyzer
from pqueens.example_simulator_functions.ma2009 import ma2009
from pqueens.example_simulator_functions.perdikaris_1dsin_lofi import perdikaris_1dsin_lofi
from pqueens.example_simulator_functions.perdikaris_1dsin_hifi import perdikaris_1dsin_hifi
from pqueens.example_simulator_functions.branin_lofi import branin_lofi
from pqueens.example_simulator_functions.branin_medfi import branin_medfi
from pqueens.example_simulator_functions.branin_hifi import branin_hifi
# from pqueens.example_simulator_functions.agawal import agawal
from pqueens.example_simulator_functions.park91b_lofi import park91b_lofi
from pqueens.example_simulator_functions.park91b_hifi import park91b_hifi
from pqueens.example_simulator_functions.park91a_lofi import park91a_lofi
from pqueens.example_simulator_functions.park91a_hifi import park91a_hifi
from pqueens.example_simulator_functions.currin88_lofi import currin88_lofi
from pqueens.example_simulator_functions.currin88_hifi import currin88_hifi
from pqueens.example_simulator_functions.borehole_lofi import borehole_lofi
from pqueens.example_simulator_functions.borehole_hifi import borehole_hifi
from pqueens.example_simulator_functions.ishigami import ishigami
from pqueens.example_simulator_functions.sobol import sobol

# test of my implementation of the Morris function
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

num_traj = 1000
num_traj_chosen = 1000
grid_jump = 2
num_levels = 4
confidence_level = 0.95
optim = False
num_bootstrap_conf = 1000

MCD = MorrisCampolongoDesigner(paramsIshigami, num_traj ,optim, num_traj_chosen, grid_jump, num_levels)
B_star, perm = MCD.get_all_samples()
Y = np.ones((num_levels,num_traj))
for i in range(num_traj):
    for j in range(num_levels):
        Y[j,i] = ishigami(B_star[i,j,0],B_star[i,j,1],B_star[i,j,2])

MA = MorrisAnalyzer(paramsIshigami,num_traj_chosen,grid_jump,num_levels,confidence_level,num_bootstrap_conf)
Si = MA.analyze(B_star, Y, perm)
MA.print_results(Si)

plt.plot(Si['mu_star'], Si['sigma'],'ro')
plt.xlabel('mu*')
plt.ylabel('Sigma')
plt.axis([0, 150, 0, 50])
plt.text(Si['mu_star'][0]+0.015, Si['sigma'][0]+0.015,  r'x1')
plt.text(Si['mu_star'][1]+0.015, Si['sigma'][1]+0.015,  r'x2')
plt.text(Si['mu_star'][2]+0.015, Si['sigma'][2]+0.015,  r'x3')
#plt.text(Si['mu_star'][3]+0.015, Si['sigma'][3]+0.015,  r'x4')
#plt.text(Si['mu_star'][4]+0.015, Si['sigma'][4]+0.015,  r'x5')
#plt.text(Si['mu_star'][5]+0.015, Si['sigma'][5]+0.015,  r'x6')
#plt.text(Si['mu_star'][6]+0.015, Si['sigma'][6]+0.015,  r'x7')
#plt.text(Si['mu_star'][7]+0.015, Si['sigma'][7]+0.015,  r'x8')
plt.title('Morris Sensitivity Indices ')
plt.show()


# Comparison with the SALib
problemBorehole= {
'num_vars': 8,
'names': ['x1', 'x2','x3','x4','x5','x6','x7','x8'],
'bounds' : [[0.05,0.15],[100,50000],[63070,115600],[990,1110],[63.1,116],[700,820],[1120,1680],[9855,12045]],
'function': borehole_hifi,
'groups': None
}

problemIshigami = {
'num_vars': 3,
'names': ['x1', 'x2','x3'],
'bounds': [[-math.pi, math.pi],[-math.pi, math.pi],[-math.pi, math.pi]],
'function': ishigami,
'groups': None
}

problemSobol = {
'num_vars': 6,
'names': ['x1', 'x2','x3','x4', 'x5','x6'],
'bounds': [[0, 1],[0, 1],[0,1],[0, 1],[0, 1],[0,1]],
'function': sobol,
'groups': None
}

num_traj = 1000

X = mrrs.sample(problemIshigami, num_traj, num_levels=4, grid_jump=2,optimal_trajectories = None, local_optimization = False)
Y = Ishigami.evaluate(X)
Si = morris.analyze(problemIshigami, X, Y, conf_level=0.95,  print_to_console=True, num_levels=4, grid_jump=2)
