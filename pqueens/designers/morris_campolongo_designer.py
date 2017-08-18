from .abstract_designer import AbstractDesigner
import numpy as np
import math
import random
from itertools import combinations
from scipy.spatial.distance import cdist

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
        self.numparams = len(params)
        self.Delta = self.grid_jump/(self.num_levels-1)
        self.scale = np.ones((1,self.numparams))
        self.bounds = np.ones((2,self.numparams))
        i = 0
        for name in params.keys():
            self.bounds[0,i] = params[name]['min']
            self.bounds[1,i] = params[name]['max']
            # If our input space is not [0 1] scale length of Delta
            self.scale[0,i] = (self.bounds[1,i]-self.bounds[0,i])
            i = i+1
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
        if np.array_equal(B_star_optim[m,:,:], B_star_optim[l,:,:]):
            distance = 0
        else:
            distance = np.array(np.sum(cdist(B_star_optim[m,:,:], B_star_optim[l,:,:])), dtype=np.float32)
        return distance

    def compute_distance_matrix(self,B_star, num_traj):
        """
        Store the distances between all pairs of trajectories

        Attributes :
        B_Star (np.array)
            Tensor which stores the coordinates of each trajectories
        num_traj (int) :
            Number of trajectories we have

        Returns :
        distance_matrix (np.array)
            Matrix of size (np.array x np.array) which indice (i,j) corresponds
            to the distance between the trajectories i and j
        """
        distance_matrix = np.zeros((num_traj,num_traj))
        for m in range(num_traj):
            for l in range(m+1,num_traj):
                distance_matrix[l,m] = self.compute_distance(B_star,m,l)**2

        return distance_matrix

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
        D_stock = np.zeros((1,len(p)))
        distance_matrix = self.compute_distance_matrix(B_star_optim, self.num_traj)
        for i in range(len(p)):
            vector_possible = p[i]
            D = 0
            for j in range(len(vector_possible)):
                for k in range(j+1,len(vector_possible)):
                    D = D + distance_matrix[vector_possible[k],vector_possible[j]]
            D = math.sqrt(D)
            D_stock[0,i]= D
        v,imax = D_stock.max(), D_stock.argmax()
        return p, imax

    def get_all_samples(self):
        """
        Function which generate the np.array B_star, either directly from Morris
        Design or with the optimization of Campolongo
        """
        # Definition of useful matrices for the computation of the trajectories
        B = np.zeros((self.numparams+1,self.numparams))
        for i in range(self.numparams+1):
            for j in range(self.numparams):
                if i > j:
                    B[i,j] = 1
        J_k = np.ones((self.numparams+1,self.numparams))
        J_1 = np.ones((self.numparams+1,1))

        B_star = np.zeros((self.numparams+1,self.numparams))
        B_star_optim =  np.zeros((self.num_traj,self.numparams+1,self.numparams))
        P_star_optim =  np.zeros((self.num_traj,self.numparams,self.numparams))
        perm_optim =  np.zeros((self.num_traj,self.numparams))
        B_star_chosen = np.ones((self.num_traj_chosen,self.numparams+1,self.numparams))
        perm_chosen =  np.ones((self.num_traj_chosen,self.numparams))

        for r in range(self.num_traj):
            D_star = np.zeros((self.numparams,self.numparams))
            for i in range(self.numparams):
                D_star[i,i] = random.choice([-1,1])

            perm = np.random.permutation(self.numparams)
            P_star = np.zeros((self.numparams,self.numparams))
            for i in range(self.numparams):
                P_star[i, perm[i]] = 1
            choices = np.zeros((2,self.numparams))
            for i in range(self.numparams):
                choices[0,i] = 0
                choices[1,i] = 1 - self.Delta
            x_star = np.zeros((1,self.numparams))
            for i in range(self.numparams):
                x_star[0,i] = random.choice(choices[:,i])
            # Computation of B_star
            B_star = np.dot(np.dot(J_1,x_star) +
            self.Delta/2*(np.dot((2*B - J_k),D_star) + J_k),P_star)

            B_temp = np.zeros((self.numparams+1,self.numparams))
            for i in range(self.numparams):
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
