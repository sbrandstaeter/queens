import numpy as np
import random
import math
from scipy.stats import norm
from pqueens.example_simulator_functions.ma2009 import ma2009
from pqueens.example_simulator_functions.perdikaris_1dsin_lofi import perdikaris_1dsin_lofi
from pqueens.example_simulator_functions.perdikaris_1dsin_hifi import perdikaris_1dsin_hifi
from pqueens.example_simulator_functions.branin_lofi import branin_lofi
from pqueens.example_simulator_functions.branin_medfi import branin_medfi
from pqueens.example_simulator_functions.branin_hifi import branin_hifi
from pqueens.example_simulator_functions.park91b_lofi import park91b_lofi
from pqueens.example_simulator_functions.park91b_hifi import park91b_hifi
from pqueens.example_simulator_functions.park91a_lofi import park91a_lofi
from pqueens.example_simulator_functions.park91a_hifi import park91a_hifi
from pqueens.example_simulator_functions.currin88_lofi import currin88_lofi
from pqueens.example_simulator_functions.currin88_hifi import currin88_hifi
from pqueens.example_simulator_functions.borehole_lofi import borehole_lofi
from pqueens.example_simulator_functions.borehole_hifi import borehole_hifi
from pqueens.example_simulator_functions.ishigami import ishigami

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
    output_samples (int):
        The number of output samples Y (default 1)


    Examples
    --------
    MCD = MorrisCampolongoDesigner(problem, num_traj ,optim, num_traj_chosen, grid_jump, num_levels)
    X, perm = MCD.get_all_samples()

    MA = MorrisAnalyzer(problem,num_traj_chosen,grid_jump,num_levels,confidence_level,num_bootstrap_conf, output_samples)
    Si = MA.analyze(X, perm)
        """

    def __init__(self,params,num_traj_chosen,grid_jump,num_levels,
                confidence_level,num_bootstrap_conf,output_samples):
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
        output_samples (int):
            The number of output samples Y (default 1)

        """
        self.params = params
        self.dim = params['num_vars']
        self.num_traj_chosen = num_traj_chosen
        self.grid_jump = grid_jump
        self.num_levels = num_levels
        self.Delta = self.grid_jump/(self.num_levels-1)
        self.confidence_level  = confidence_level
        self.num_bootstrap_conf = num_bootstrap_conf
        self.output_samples = output_samples

    def compute_elementary_effect(self, B_star, perm):
        """ Function to compute the elementary effect for the different
        functions of the framework pqueens.
        Args:
            B_star (np.array): np.array with the input space design from [1]
            perm (np.array): np.array associated to B_star which corresponds
            to the order in which factors are moved.
        """
        EE = np.ones((1,self.dim), dtype = float)
        for i in range(self.dim):
            s = np.sign(B_star[i+1,perm[i]]-B_star[i,perm[i]])
            if self.params['function'] == ma2009:
                num = (ma2009(B_star[i+1,0],B_star[i+1,1])
                -ma2009(B_star[i,0],B_star[i,1]))
                den = s*self.Delta
                EE[0,perm[i]] = num / den
            if self.params['function'] == perdikaris_1dsin_lofi:
                num = (perdikaris_1dsin_lofi(B_star[i+1,0])
                -perdikaris_1dsin_lofi(B_star[i,0]))
                den = s*self.Delta
                EE[0,perm[i]] = num / den
            if self.params['function'] == perdikaris_1dsin_hifi:
                num = (perdikaris_1dsin_hifi(B_star[i+1,0])
                -perdikaris_1dsin_hifi(B_star[i,0]))
                den = s*self.Delta
                EE[0,perm[i]] = num / den
            if self.params['function'] == branin_lofi:
                num = (branin_lofi(B_star[i+1,0],B_star[i+1,1])
                -branin_lofi(B_star[i,0],B_star[i,1]))
                den = s*self.Delta
                EE[0,perm[i]] = num / den
            if self.params['function'] == branin_medfi:
                num = (branin_medfi(B_star[i+1,0],B_star[i+1,1])
                -branin_medfi(B_star[i,0],B_star[i,1]))
                den = s*self.Delta
                EE[0,perm[i]] = num / den
            if self.params['function'] == branin_hifi:
                num = (branin_hifi(B_star[i+1,0],B_star[i+1,1])
                -branin_hifi(B_star[i,0],B_star[i,1]))
                den = s*self.Delta
                EE[0,perm[i]] = num / den
            if self.params['function'] == park91b_lofi:
                num = (park91b_lofi(B_star[i+1,0],B_star[i+1,1],
                B_star[i+1,2],B_star[i+1,3])-park91b_lofi(B_star[i,0],
                B_star[i,1],B_star[i,2],B_star[i,3]))
                den = s*self.Delta
                EE[0,perm[i]] = num / den
            if self.params['function'] == park91b_hifi:
                num = (park91b_hifi(B_star[i+1,0],B_star[i+1,1],
                B_star[i+1,2],B_star[i+1,3])-park91b_hifi(B_star[i,0],
                B_star[i,1],B_star[i,2],B_star[i,3]))
                den = s*self.Delta
                EE[0,perm[i]] = num / den
            if self.params['function'] == park91a_lofi:
                num = (park91a_lofi(B_star[i+1,0],B_star[i+1,1],
                B_star[i+1,2],B_star[i+1,3])-park91a_lofi(B_star[i,0],
                B_star[i,1],B_star[i,2],B_star[i,3]))
                den = s*self.Delta
                EE[0,perm[i]] = num / den
            if self.params['function'] == park91a_hifi:
                num = (park91a_hifi(B_star[i+1,0],B_star[i+1,1],
                B_star[i+1,2],B_star[i+1,3])-park91a_hifi(B_star[i,0],
                B_star[i,1],B_star[i,2],B_star[i,3]))
                den = s*self.Delta
                EE[0,perm[i]] = num / den
            if self.params['function'] == currin88_lofi:
                num = (currin88_lofi(B_star[i+1,0],B_star[i+1,1])
                -currin88_lofi(B_star[i,0],B_star[i,1]))
                den = s*self.Delta
                EE[0,perm[i]] = num / den
            if self.params['function'] == currin88_hifi:
                num = (currin88_hifi(B_star[i+1,0],B_star[i+1,1])
                -currin88_hifi(B_star[i,0],B_star[i,1]))
                den = s*self.Delta
                EE[0,perm[i]] = num / den
            if self.params['function'] == borehole_lofi:
                num = (borehole_lofi(B_star[i+1,0],B_star[i+1,1],B_star[i+1,2],
                B_star[i+1,3],B_star[i+1,4],B_star[i+1,5],B_star[i+1,6],B_star[i+1,7])
                -borehole_lofi(B_star[i,0],B_star[i,1],B_star[i,2],B_star[i,3],
                B_star[i,4],B_star[i,5],B_star[i,6],B_star[i,7]))
                den = s*self.Delta
                EE[0,perm[i]] = num / den
            if self.params['function'] == borehole_hifi:
                num = (borehole_hifi(B_star[i+1,0],B_star[i+1,1],B_star[i+1,2],
                B_star[i+1,3],B_star[i+1,4],B_star[i+1,5],B_star[i+1,6],B_star[i+1,7])
                -borehole_hifi(B_star[i,0],B_star[i,1],B_star[i,2],B_star[i,3],
                B_star[i,4],B_star[i,5],B_star[i,6],B_star[i,7]))
                den = s*self.Delta
                EE[0,perm[i]] = num / den
            if self.params['function'] == ishigami:
                num = (ishigami(B_star[i+1,0],B_star[i+1,1],B_star[i+1,2])
                -ishigami(B_star[i,0],B_star[i,1],B_star[i,2]))
                den = s*self.Delta
                EE[0,perm[i]] = num / den
        return EE

    def compute_confidence_interval(self,conf_level, EET, num_traj_chosen,
                                    num_bootstrap_conf):
        """Function to compute the confidence intervals for our sensitivity
        indice mu_star, function inspired from compute_mu_star_confidence from
        SALib"""
        EET_bootstrap = np.zeros([num_traj_chosen])
        data_bootstrap = np.zeros([num_bootstrap_conf])
        if not 0 < conf_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        bootstrap_index = np.random.randint(len(EET), size = (num_traj_chosen,num_bootstrap_conf))
        EET_bootstrap= EET[bootstrap_index]
        data_bootstrap = np.average(np.abs(EET_bootstrap), axis = 1)
        return norm.ppf(0.5 + conf_level/2)*data_bootstrap.std(ddof = 1)

    def analyze(self, B_star_chosen, perm_chosen):
        """Function to compute the sensitivity indices thanks to the Morris Method,
        as explained in [3] the trajectories are chosen to optimized the input space """
        EET = np.ones((self.num_traj_chosen,self.dim), dtype = float)
        for r in range(self.num_traj_chosen):
            EET[r,:] = self.compute_elementary_effect(B_star_chosen[r,:,:], perm_chosen[r,:])
        EE = np.ones((1,self.dim), dtype = float)
        for i in range(self.dim):
            EE[0,i] = np.mean(abs(EET[:,i]), axis = 0)

        Si = dict((k, [None] * self.dim)
        for k in ['names', 'mu', 'mu_star', 'sigma', 'mu_star_conf'])
        Si['mu'] = np.average(EET, 0)
        Si['mu_star'] = np.average(np.abs(EET), 0)
        Si['sigma'] = np.std(EET,axis=0)
        Si['names'] = self.params['names']
        for j in range(self.dim):
            Si['mu_star_conf'][j] = self.compute_confidence_interval(self.confidence_level,
            EET[:,j], self.num_traj_chosen, self.num_bootstrap_conf)
        print("{0:<30} {1:>10} {2:>10} {3:>15} {4:>10}".format(
                "Parameter",
                "Mu_Star",
                "Mu",
                "Mu_Star_Conf",
                "Sigma")
                )
        for j in list(range(self.dim)):
            print("{0!s:30} {1!s:10} {2!s:10} {3!s:15} {4!s:10}".format(
                    Si['names'][j],
                    Si['mu_star'][j],
                    Si['mu'][j],
                    Si['mu_star_conf'][j],
                    Si['sigma'][j])
                    )
        return Si
