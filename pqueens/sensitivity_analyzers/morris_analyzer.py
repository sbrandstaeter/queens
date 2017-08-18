import numpy as np
import random
import math
from scipy.stats import norm

class MorrisAnalyzer(object):
    """ Class to compute the sensitivity indices thanks to the Elementary Effect
    Method from Morris improved by Campolongo. The class should be run with the
    Designer MorrisCampolongoDesigner which generates the appropriate design for
    these sensitivity indices.

    References :
    [1] A. Saltelli, M. Ratto, T. Andres, F. Campolongo, T. Cariboni, D. Galtelli,
        M. Saisana, S. Tarantola. "GLOBAL SENSITIVITY ANALYSIS. The Primer",
        109 - 121,
        doi:

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

    def __init__(self,params,num_traj_chosen,grid_jump,num_levels,
                confidence_level, num_bootstrap_conf):
        """
        Args:
        params (dict) :
            The problem definition, including, num_vars (the number of variables),
            names (the names of the variables), bounds (the bounds of the input
            space), and function (the name of the function we use to compute the
            sensitivity indices)
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
        self.params = params
        self.numparams = len(params)
        self.num_traj_chosen = num_traj_chosen
        self.grid_jump = grid_jump
        self.num_levels = num_levels
        self.Delta = self.grid_jump/(self.num_levels-1)
        self.confidence_level  = confidence_level
        self.num_bootstrap_conf = num_bootstrap_conf

    def analyze(self, B_star_chosen, Y_chosen, perm_chosen):
        """Compute the sensitivity indices thanks to the Morris Method,
        The trajectories are chosen to optimized the input space as explained
        in [3]
        Args:
        B_star_chosen (np.array):
            np.array with the chosen trajectories which are defined in [1]
        Y_chosen (np.array) :
            np.array with the evaluation of the chosen trajectories by the
            function of our problem
        perm_chosen (np.array):
            np.array associated to B_star_chosen which corresponds to the order
            in which factors are moved
        Returns :
        Si (dict) :
            dictionnary with the sensitivity indices
        """
        EET = np.ones((self.num_traj_chosen,self.numparams))
        for r in range(self.num_traj_chosen):
            EET[r,:] = self.compute_elementary_effect(B_star_chosen[r,:,:], Y_chosen[:,r], perm_chosen[r,:])
        EE = np.ones((1,self.numparams))
        for i in range(self.numparams):
            EE[0,i] = np.mean(abs(EET[:,i]), axis = 0)

        Si = dict((k, [None] * self.numparams)
        for k in ['names', 'mu', 'mu_star', 'sigma', 'mu_star_conf'])
        Si['mu'] = np.average(EET, 0)
        Si['mu_star'] = np.average(np.abs(EET), 0)
        Si['sigma'] = np.std(EET,axis=0, ddof = 1)
        j = 0
        for name in self.params.keys():
            Si['names'][j] = name
            j = j + 1
        for j in range(self.numparams):
            Si['mu_star_conf'][j] = self.compute_confidence_interval(self.confidence_level,
            EET[:,j], self.num_traj_chosen,self.num_bootstrap_conf)
        return Si

    def compute_elementary_effect(self, B_star, Y, perm):
        """ Function to compute the elementary effect for the different
        functions of the framework pqueens.
        Args:
        B_star (np.array) :
            np.array with the trajectories defined in [1]
        Y (np.array) :
            np.array with the evaluation of the trajectories by the function
            of our problem
        perm (np.array) :
            np.array associated to B_star which corresponds to the order
            in which factors are moved.
        """
        EE = np.ones((1,self.numparams))
        for i in range(self.numparams):
            s = np.sign(B_star[i+1,perm[i]]-B_star[i,perm[i]])
            num = Y[i+1]-Y[i]
            den = s*self.Delta
            EE[0,perm[i]] = num / den
        return EE

    def compute_confidence_interval(self,conf_level, EET, num_traj_chosen, num_bootstrap_conf):
        """ Compute the confidence intervals for sensitivity indice mu_star.
        Function inspired from compute_mu_star_confidence from SALib
        Args :
        conf_level (float) :
            value of our confidence level (should be between 0 and 1)
        EET (np.array) :
            np.array with the values of the elementary effects for each
            trajectory
        num_traj_chosen (int) :
            Number of trajectories chosen in the design, with the brute-force
            optimization from Campolongo.
        """
        EET_bootstrap = np.zeros([num_traj_chosen])
        data_bootstrap = np.zeros([num_bootstrap_conf])
        if not 0 < conf_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        bootstrap_index = np.random.randint(len(EET), size = (num_bootstrap_conf,num_traj_chosen))
        EET_bootstrap= EET[bootstrap_index]
        data_bootstrap = np.average(np.abs(EET_bootstrap), axis = 1)
        return norm.ppf(0.5 + conf_level/2)*data_bootstrap.std(ddof = 1)

    def print_results(self, Si):
        """ Function to print the results to the the console of our sensitivity
        analysis
        Args :
        Si (dict) :
            dictionnary with the results of the sensitivity analysis
        """
        print("{0:<30} {1:>10} {2:>10} {3:>15} {4:>10}".format(
                "Parameter",
                "Mu_Star",
                "Mu",
                "Mu_Star_Conf",
                "Sigma")
                )
        for j in list(range(self.numparams)):
            print("{0!s:30} {1!s:10} {2!s:10} {3!s:15} {4!s:10}".format(
                    Si['names'][j],
                    Si['mu_star'][j],
                    Si['mu'][j],
                    Si['mu_star_conf'][j],
                    Si['sigma'][j])
                    )
