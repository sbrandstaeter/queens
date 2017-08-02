from .abstract_designer import AbstractDesigner
import numpy as np
import math
import random
from itertools import combinations

class MorrisCampolongoDesigner(object):
    """ Class to generate the Morris Design, necessary to perform sensitivity
    analysis as Morris in [2] and with Campolongo optimization in [3].

    References :
    [1] A. Saltelli, M. Ratto, T. Andres, F. Campolongo, T. Cariboni, D. Galtelli,
        M. Saisana, S. Tarantola. "GLOBAL SENSITIVITY ANALYSIS. The Primer",
        109 - 121,
        ISBN : 978-0-470-05997-5

    [2] Morris, M. (1991).  "Factorial Sampling Plans for Preliminary
        Computational Experiments."  Technometrics, 33(2):161-174,
        doi:10.1080/00401706.1991.10484804.

    [3] Campolongo, F., J. Cariboni, and A. Saltelli (2007).  "An effective
        screening design for sensitivity analysis of large models."
        Environmental Modelling & Software, 22(10):1509-1518,
        doi:10.1016/j.envsoft.2006.10.004.

    Attributes:

        params (dict) :
            The problem definition
        num_traj (int) :
            Number of trajectories in the input space
        optim (bool) :
            True if we want to perform brutal-force optimization from Campolongo
            False if we choose directly the num_traj, in this case num_traj and
            num_traj_chosen have to be equal
        num_traj_chosen (int) :
            Number of trajectories chosen in the design, with the brute-force
            optimization from Campolongo.
        grid_jump (int) :
            The grid jump size, must be identical to the value passed
            to :class :`morris_campolongo_designer` (default 2)
        num_levels (int) :
            The number of grid levels, must be identical to the value
            passed to `morris_campolongo_designer` (default 4)
        confidence_level (float) :
            The confidence interval level (default 0.95)
        num_bootstrap_conf (int) :
            Number of bootstrap iterations for the computation of confidence intervals
        output_samples (int):
            The number of output samples Y (default 1)

    Returns:
        B_star_chosen : numpy.array
            Numpy array with the trajectories
        perm : numy.array
            Numpy.array with the order in which the factors are moved
    """

    def __init__(self,params,num_traj, optim, num_traj_chosen, grid_jump, num_levels):
        """
        Args:
        params (dict) :
            The problem definition
        num_traj (int) :
            Number of trajectories in the input space
        optim (bool) :
            True if we want to perform brutal-force optimization from Campolongo
            False if we choose directly the num_traj, in this case num_traj and
            num_traj_chosen have to be equal
        num_traj_chosen (int) :
            Number of trajectories chosen in the design, with the brute-force
            optimization from Campolongo.
        grid_jump (int) :
            The grid jump size, must be identical to the value passed
            to :class :`morris_campolongo_designer` (default 2)
        num_levels (int) :
            The number of grid levels, must be identical to the value
            passed to `morris_campolongo_designer` (default 4)
        confidence_level (float) :
            The confidence interval level (default 0.95)
        num_bootstrap_conf (int) :
            Number of bootstrap iterations for the computation of confidence intervals

        """
        self.num_traj = num_traj
        self.optim = optim
        self.num_traj_chosen = num_traj_chosen
        self.grid_jump = grid_jump
        self.num_levels = num_levels
        self.dim = params['num_vars']
        self.Delta = self.grid_jump/(self.num_levels-1)
        self.scale = np.ones((1,self.dim), dtype = float)
        self.bounds = np.ones((2,self.dim))
        for i in range(self.dim):
            self.bounds[0,i] = params['bounds'][i][0]
            self.bounds[1,i] = params['bounds'][i][1]
            # If our input space is not [0 1] scale length of Delta
            self.scale[0,i] = (self.bounds[1,i]-self.bounds[0,i])
    def compute_distance(self,B_star_optim,m,l):
        """
        Function to compute the distance between a pair of trajectories m
        and l, as in equation (3.3) from [1]

        Attributes :
        B_Star_optim (np.array)
            Tensor which stores the coordinates of each trajectories
        m, l (int) :
            Number of the trajectories between which we want to
            calculate the distance
        """
        if m == l:
            d_ml = 0
        else:
            s = 0
            for i in range(self.dim+1):
                for j in range(self.dim+1):
                    d = math.sqrt(np.sum(B_star_optim[m,i,:]-B_star_optim[m,j,:])**2)
                    s = s + d
            d_ml = s
        return d_ml

    # Choice of the best trajectories:
    def choose_best_trajectory(self,B_star_optim):
        """
        Function to choose the trajectories to maximize their spread in the
        input space.

        Attributes :
        B_Star_optim (np.array)
            Tensor which stores the coordinates of each trajectories
        num_traj (int) :
            Number of trajectories in the input space
        num_traj_chosen (int) :
            Number of trajectories chosen in the design, with the brute-force
            optimization from Campolongo.
        """
        p = combinations(range(self.num_traj),self.num_traj_chosen)
        nb_combi = (math.factorial(self.num_traj)//(math.factorial(self.num_traj_chosen)*
        math.factorial(self.num_traj-self.num_traj_chosen)))
        p = np.zeros((nb_combi,self.num_traj_chosen),dtype = int)
        ind = 0
        for subset in combinations(range(self.num_traj), self.num_traj_chosen):
            p[ind] = np.asarray(subset)
            ind = ind+1
        D_stock = np.zeros((1,len(p)), dtype = float)
        for i in range(len(p)):
            vector_possible = p[i]
            #print('vector_possible')
            #print(vector_possible)
            D = 0
            for j in vector_possible:
                for k in vector_possible:
                    temp = self.compute_distance(B_star_optim,j,k)
                    D = D + self.compute_distance(B_star_optim,j,k)**2
                    D = math.sqrt(D)
            D_stock[0,i]= D
        # We are looking for the r maximums in D
        v,imax = D_stock.max(), D_stock.argmax()
        return p, imax

    def get_all_samples(self):
        """
        Function which generate the np.array B_star, either directly from Morris
        Design or with the optimization of Campolongo
        """
        # Definition of useful matrices for the computation of the trajectories
        B = np.zeros((self.dim+1,self.dim), dtype = float)
        for i in range(self.dim+1):
            for j in range(self.dim):
                if i > j:
                    B[i,j] = 1
        J_k = np.ones((self.dim+1,self.dim), dtype = float)
        J_1 = np.ones((self.dim+1,1), dtype = float)

        B_star = np.zeros((self.dim+1,self.dim), dtype = float)
        B_star_optim =  np.zeros((self.num_traj,self.dim+1,self.dim), dtype = float)
        P_star_optim =  np.zeros((self.num_traj,self.dim,self.dim), dtype = float)
        perm_optim =  np.zeros((self.num_traj,self.dim), dtype = float)
        B_star_chosen = np.ones((self.num_traj_chosen,self.dim+1,self.dim), dtype = float)
        perm_chosen =  np.ones((self.num_traj_chosen,self.dim), dtype = float)

        for r in range(self.num_traj):
            D_star = np.zeros((self.dim,self.dim), dtype = float)
            for i in range(self.dim):
                D_star[i,i] = random.choice([-1,1])

            perm = np.random.permutation(self.dim)
            P_star = np.zeros((self.dim,self.dim), dtype = float)
            for i in range(self.dim):
                P_star[i, perm[i]] = 1
            choices = np.zeros((2,self.dim), dtype = float)
            for i in range(self.dim):
                choices[0,i] = 0
                choices[1,i] = 1 - self.Delta
            x_star = np.zeros((1,self.dim), dtype = float)
            for i in range(self.dim):
                x_star[0,i] = random.choice(choices[:,i])
            # Computation of B_star
            B_star = np.dot(np.dot(J_1,x_star) +
            self.Delta/2*(np.dot((2*B - J_k),D_star) + J_k),P_star)

            B_temp = np.zeros((self.dim+1,self.dim), dtype = float)
            for i in range(self.dim):
                B_temp[:,i] = self.bounds[0,i]
            B_star = self.scale*B_star+B_temp
            B_star_optim[r,:,:] = B_star
            P_star_optim[r,:,:] = P_star
            perm_optim[r,:] = perm
            perm_optim = perm_optim.astype(int)

        if self.optim == True:
            p, i = self.choose_best_trajectory(B_star_optim)
            B_star_chosen = B_star_optim[p[i],:,:]
            perm_chosen = perm_optim[p[i],:]
            perm_chosen = perm_chosen.astype(int)
            return B_star_chosen, perm_chosen
        if self.optim == False:
            return B_star_optim, perm_optim
