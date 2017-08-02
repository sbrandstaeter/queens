from pqueens.designers.monte_carlo_designer import MonteCarloDesigner
from pqueens.designers.lhs_designer import LatinHyperCubeDesigner
from pqueens.designers.pseudo_saltelli_designer import PseudoSaltelliDesigner
from pqueens.designers.morris_campolongo_designer import MorrisCampolongoDesigner
from pqueens.designers.group_designer import GroupDesigner
from SALib.sample import morris as mrrs
from SALib.test_functions import Ishigami
from SALib.analyze import morris
from SALib.analyze import sobol
from SALib.sample import saltelli
import numpy as np
import math
from pyDOE import lhs
import random
seed=43
num_samples=5

# params = { 'X' :{'x1' : np.random.uniform(-math.pi, math.pi,num_samples), 'x2' : np.random.uniform(-math.pi, math.pi,num_samples),'x3' : np.random.uniform(-math.pi, math.pi,num_samples)},
# 'X_tilde' :{'x1' : np.random.uniform(-math.pi, math.pi,num_samples), 'x2' : np.random.uniform(-math.pi, math.pi,num_samples),'x3' : np.random.uniform(-math.pi, math.pi,num_samples)}}

# GD = GroupDesigner(params,num_samples)
# Y = GD.get_all_samples()
# print(Y)

params = {
'num_vars': 3,
'names': ['x1', 'x2', 'x3'],
'bounds': [[-math.pi, math.pi]]*3
}
num_traj = 10
num_traj_chosen = 4
optim = True
grid_jump = 2
nums_levels = 4
dim = 3
X = mrrs.sample(params, num_traj, num_levels=4, grid_jump=2,optimal_trajectories = None, local_optimization = False)
#print(X)
MCD = MorrisCampolongoDesigner(params,num_traj, optim, num_traj_chosen, grid_jump, nums_levels)

Y = MCD.get_all_samples()
#print(Y)
