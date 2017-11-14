import math
import random
from itertools import combinations
import scipy.spatial.distance
import numpy as np
from pqueens.models.model import Model
from pqueens.variables.variables import Variables
from .iterator import Iterator
from scipy.stats import norm


# from .abstract_designer import AbstractDesigner

class MorrisCampolongoIterator(Iterator):
    """ Morris Campolongo Iterator to compute elementary effects

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

        num_traj (int):             Number of trajectories in the input space

        optim (bool):               True if we want to perform brute-force
                                    optimization from Campolongo. False if we
                                    choose directly the num_traj, in this case
                                    num_traj and num_traj_chosen have to be equal

        num_traj_chosen (int):      Number of trajectories chosen in the design,
                                    with the brute-force optimization from Campolongo.

        grid_jump (int):            The grid jump size

        num_levels (int):           The number of grid levels

        num_samples (int):          The number of samples

        seed (int):                 Seed for random number generation
    """
    def __init__(self, model, num_traj, optim, num_traj_chosen, grid_jump,
                 num_levels, seed, num_samples, confidence_level, num_bootstrap_conf):
        super(MorrisCampolongoIterator, self).__init__(model)
        """ Initialize MorrisCampolongoIterator

        Args:
            num_traj (int):         Number of trajectories in the input space

            optim (bool):           True if we want to perform brutal-force
                                    optimization from Campolongo. False if we
                                    choose directly the num_traj, in this case
                                    num_traj and num_traj_chosen have to be equal

            num_traj_chosen (int):  Number of trajectories chosen in the design,
                                    with the brute-force optimization from Campolongo
            grid_jump (int):        The grid jump size

            num_levels (int):       The number of grid levels

            seed (int):             Seed for random number generation

            num_samples (int):      Number of samples

            samples (np.array):      Samples
            perm (???):              ???
            Delta (???):             ???

            self.confidence_level (float):
            self.num_bootstrap_conf (float):

        """
        self.num_traj = num_traj
        self.optim = optim
        self.num_traj_chosen = num_traj_chosen
        self.grid_jump = grid_jump
        self.num_levels = num_levels
        self.seed = seed
        self.num_samples = num_samples
        self.samples = None
        # TODO define this
        self.perm = None

        # TODO add docstring
        self.Delta = self.grid_jump/(self.num_levels-1)
        self.confidence_level  = confidence_level
        self.num_bootstrap_conf = num_bootstrap_conf

    @classmethod
    def from_config_create_iterator(cls, config):
        """ Create MorrisCampolongo iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description

        Returns:
            iterator: MorrisCampolongo iterator object

        """
        method_options = config["method"]["method_options"]
        model_name = method_options["model"]

        model = Model.from_config_create_model(model_name, config)

        return cls(model, method_options["num_traj"], method_options["optim"],
                   method_options["num_traj_chosen"], method_options["grid_jump"],
                   method_options["number_of_levels"], method_options["seed"],
                   method_options["num_samples"], method_options["confidence_level"],
                   method_options["num_bootstrap_conf"])

    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()

    def pre_run(self):
        """ Generate samples for subsequent analysis and update model """
        np.random.seed(self.seed)

        distribution_info = self.model.get_parameter_distribution_info()
        numparams = len(distribution_info)
        # fix this
        self.numparams = numparams
        scale = np.ones((1, numparams))
        bounds = np.ones((2, numparams))
        i = 0
        for distribution in distribution_info:
            bounds[0, i] = distribution['min']
            bounds[1, i] = distribution['max']
            # If our input space is not [0 1] scale length of the different ranges
            scale[0, i] = (bounds[1, i]-bounds[0, i])
            i = i+1

        #self.samples = np.zeros((self.num_samples, numparams))

        random.seed(self.seed)
        np.random.seed(self.seed)

        # Definition of useful matrices for the computation of the trajectories
        # initialisation of B a lower trinagular matrix with only ones
        B = np.zeros((numparams+1, numparams))
        for i in range(numparams+1):
            for j in range(numparams):
                if i > j:
                    B[i,j] = 1
        # initialisation of J_k a (k+1) x k matrix of ones
        J_k = np.ones((numparams+1, numparams))
        # initialisation of J_k a (k+1) x 1 matrix of ones
        J_1 = np.ones((numparams+1, 1))
        # initialisation of the matrix B_star as defined in [2], which will stores
        # the coordinates of our different points of our trajectories
        B_star = np.zeros((numparams+1, numparams))
        # matrix B_star in case no
        B_star_optim = np.zeros((self.num_traj, numparams+1, numparams))
        perm_optim = np.zeros((self.num_traj, numparams))
        # initialisation of the matrix B_star where only the trajectories chosen
        # with the brute-force strategy are kept
        B_star_chosen = np.ones((self.num_traj_chosen, numparams+1, numparams))
        # initialisation of the vector perm_chosen where only the permutation
        # indices of the trajectories chosen with the brute-force strategy are kept
        perm_chosen = np.ones((self.num_traj_chosen, numparams))

        # loop over each trajectory
        for r in range(self.num_traj):
            # defintion of D_star a d-dimensional diagonal matrix in which
            # each element is either -1 or 1 with equal probability
            D_star = np.zeros((numparams, numparams))
            for i in range(numparams):
                D_star[i, i] = random.choice([-1, 1])

            # definition of P_star the permutation matrix which gives the order
            # in which the factors are moved
            perm = np.random.permutation(numparams)
            P_star = np.zeros((numparams, numparams))
            for i in range(numparams):
                P_star[i, perm[i]] = 1
            # initialisation of the first point of the trajectory x_star
            choices = np.zeros((2, numparams))
            for i in range(numparams):
                choices[0, i] = 0
                choices[1, i] = 1 - self.Delta
            x_star = np.zeros((1, numparams))
            for i in range(numparams):
                x_star[0, i] = random.choice(choices[:, i])
            # Computation of B_star
            B_star = np.dot(np.dot(J_1, x_star) +
                            self.Delta/2*(np.dot((2*B - J_k), D_star) + J_k), P_star)
            # rescaling of the B_star matrix in case where the ranges of the input
            # factors are not [0,1]
            B_temp = np.zeros((numparams+1, numparams))
            for i in range(numparams):
                B_temp[:, i] = bounds[0, i]
            B_star = scale*B_star+B_temp
            # store the different trajectories in a 3-dimensional np.array and
            # also the order the factors are moved
            B_star_optim[r, :, :] = B_star
            perm_optim[r, :] = perm
            perm_optim = perm_optim.astype(int)

        if self.optim:
            # in this case num_traj_chosen trajectories are chosen among the
            # num_traj with the brute force strategy from Campolongo
            p, i = self.__choose_best_trajectory(B_star_optim)
            B_star_chosen = B_star_optim[p[i], :, :]
            perm_chosen = perm_optim[p[i], :]
            perm_chosen = perm_chosen.astype(int)
            self.samples = B_star_chosen
            self.perm = perm_chosen

        if not self.optim:
            # in this case we do not choose any trjectory, we directly take all
            # trajectories
            self.samples = B_star_optim
            self.perm = perm_optim

    def core_run(self):
        """  Run Analysis on model """

        inputs = np.reshape(self.samples,(-1,8))
        #print("shape of inputs {}".format(inputs.shape))
        self.model.update_model_from_sample_batch(inputs)
        outputs = self.eval_model()
        print("ouputs {}".format(np.reshape(outputs,(9,4),order='F')))

        Si = self.__analyze(self.samples, np.reshape(outputs,(9,4), order='F'), self.perm)
        S = self.__print_results(Si)
    def post_run(self):
        """ Analyze the results """



    def __compute_distance(self, B_star_optim, m, l):
        """ Compute distance between a pair of trajectories m and l

        Args:
            B_Star_optim (np.array): Tensor which stores the coordinates of
                                     each trajectories
            m (int):                 Index of first trajectory
            l (int):                 Index of second trajectory

        Returns:
            float: distance between trajectory m and l
        """
        if np.array_equal(B_star_optim[m, :, :], B_star_optim[l, :, :]):
            distance = 0
        else:
            # cdist computes distance between each pair of the collections of inputs
            distance = np.array(np.sum(scipy.spatial.distance.cdist(B_star_optim[m, :, :],
                                                                    B_star_optim[l, :, :])), dtype=np.float32)
        return distance

    def __compute_distance_matrix(self, B_star, num_traj):
        """ Store the distances between all pairs of trajectories

        Args:

            B_Star (np.array):  Tensor which stores the coordinates of each trajectories
            num_traj (int):     Number of trajectories we have

        Returns :
            np.array:           Matrix of size (np.array x np.array) which
                                indice (i,j) corresponds to the distance between
                                the trajectories i and j
        """
        # initialisation of the matrix to store all the distances between the trjectories
        distance_matrix = np.zeros((num_traj, num_traj))
        # Fill the matrix
        for m in range(num_traj):
            for l in range(m+1, num_traj):
                distance_matrix[l, m] = self.__compute_distance(B_star, m, l)**2
        return distance_matrix


    def __choose_best_trajectory(self, B_star_optim):
        """ Choose the trajectories to maximize their spread in the input space.

        Args:
            B_Star_optim (np.array): Tensor which stores the coordinates of
                                     each trajectories
            num_traj (int):          Number of trajectories in the input space
            num_traj_chosen (int):   Number of trajectories chosen in the design,
                                     with the brute-force optimization from Campolongo.

        Returns:
            ???
            # TODO find out return type
        """
        # creation of a vector p containing all possible indices combinations for
        # the trajectories
        nb_combi = (math.factorial(self.num_traj)//(math.factorial(self.num_traj_chosen)*
                                                    math.factorial(self.num_traj-self.num_traj_chosen)))
        p = np.zeros((nb_combi, self.num_traj_chosen), dtype = int)
        ind = 0
        for subset in combinations(range(self.num_traj), self.num_traj_chosen):
            p[ind] = np.asarray(subset)
            ind = ind+1
        # we are looking for the combination which has the maximal distance D as
        # defined in [3]
        # Generate the matrix with all distances between two pairs of combinations
        distance_matrix = self.__compute_distance_matrix(B_star_optim, self.num_traj)
        # D_stock stores all possible values of D
        D_stock = np.zeros((1, len(p)))
        # Compute all possible D for each combinations
        for i in range(len(p)):
            vector_possible = p[i]
            D = 0
            for j in range(len(vector_possible)):
                for k in range(j+1, len(vector_possible)):
                    D = D + distance_matrix[vector_possible[k], vector_possible[j]]
            D = math.sqrt(D)
            D_stock[0, i] = D
        _, imax = D_stock.max(), D_stock.argmax()
        # return p[imax] the combination with the maximal D
        return p, imax

    def __analyze(self, B_star_chosen, Y_chosen, perm_chosen):
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
        # computation of the distribution of all elementary effects
        EET = np.ones((self.num_traj_chosen,self.numparams))
        for r in range(self.num_traj_chosen):
            EET[r,:] = self.__compute_elementary_effect(B_star_chosen[r,:,:], Y_chosen[:,r], perm_chosen[r,:])

        # creation of the dictionnary to store all the results
        Si = dict((k, [None] * self.numparams)
        for k in ['names', 'mu', 'mu_star', 'sigma', 'mu_star_conf'])
        Si['mu'] = np.average(EET, 0)
        Si['mu_star'] = np.average(np.abs(EET), 0)
        Si['sigma'] = np.std(EET,axis=0, ddof = 1)
        #j = 0
        #for name in self.params.keys():
        for j in range(self.numparams):
            Si['names'][j] = j
            #j = j + 1
        for j in range(self.numparams):
            Si['mu_star_conf'][j] = self.__compute_confidence_interval(self.confidence_level,
            EET[:,j], self.num_traj_chosen,self.num_bootstrap_conf)
        return Si

    def __compute_elementary_effect(self, B_star, Y, perm):
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
        EE = np.ones((1, self.numparams))
        for i in range(self.numparams):
            s = np.sign(B_star[i+1, perm[i]]-B_star[i, perm[i]])
            num = Y[i+1]-Y[i]
            den = s*self.Delta
            EE[0,perm[i]] = num / den
        return EE

    def __compute_confidence_interval(self, conf_level, EET, num_traj_chosen, num_bootstrap_conf):
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

    def __print_results(self, Si):
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
            "Sigma"))

        for j in list(range(self.numparams)):
            print("{0!s:30} {1!s:10} {2!s:10} {3!s:15} {4!s:10}".format(
                Si['names'][j],
                Si['mu_star'][j],
                Si['mu'][j],
                Si['mu_star_conf'][j],
                Si['sigma'][j]))
