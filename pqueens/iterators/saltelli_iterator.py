import numpy as np
import math
from pqueens.models.model import Model
from .iterator import Iterator
from itertools import combinations
# import random
from scipy.stats import norm
from scipy import stats


class SaltelliIterator(Iterator):
    """ Pseudo Saltelli iterator

        The purpose of this class is to generate the design neccessary to compute
        the sensitivity indices according to Le Gratiet method presented in [1],
        with the two samples X and X_tilde.

    References:

    [1] Le Gratiet, L., Cannamela, C., & Iooss, B. (2014).
        "A Bayesian Approach for Global Sensitivity Analysis
        of (Multifidelity) Computer Codes." SIAM/ASA Uncertainty
        Quantification Vol. 2. pp. 336-363,
        doi:10.1137/13926869

    [2] Sobol, I. M. (2001).  "Global sensitivity indices for nonlinear
        mathematical models and their Monte Carlo estimates."  Mathematics
        and Computers in Simulation, 55(1-3):271-280,
        doi:10.1016/S0378-4754(00)00270-6.

    Attributes:
        seed (int):                     Seed for random number generator
        num_samples (int):              Number of samples
        samples (np.array):             Array with all samples
        calc_second_order (bool):       Calculate second-order sensitivities
        num_bootstrap_samples (int):    Number of bootstrap samples
        confidence_level (float):       The confidence interval level
    """
    def __init__(self, model, seed, num_samples, calc_second_order,
                 num_bootstrap_samples, confidence_level):
        """ Initialize Saltelli iterator object

        Args:
            seed (int):                     Seed for random number generation
            num_samples (int):              Number of desired (random) samples
            calc_second_order (bool):       Calculate second-order sensitivities
            num_bootstrap_samples (int):    Number of bootstrap samples
            confidence_level (float):       The confidence interval level
        """
        super(SaltelliIterator, self).__init__(model)

        self.num_samples = num_samples
        self.seed = seed
        self.calc_second_order = calc_second_order
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level = confidence_level

        self.samples = None

        self.output_samples = 1

    @classmethod
    def from_config_create_iterator(cls, config):
        """ Create Saltelli iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description

        Returns:
            iterator: SaltelliIterator object

        """
        method_options = config["method"]["method_options"]
        model_name = method_options["model"]

        model = Model.from_config_create_model(model_name, config)
        return cls(model, method_options["seed"],
                   method_options["num_samples"],
                   method_options["calc_second_order"],
                   method_options["num_bootstrap_samples"],
                   method_options["confidence_level"])

    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()

    def pre_run(self):
        """ Generate samples for subsequent analysis and update model """
        # fix seed of random number generator
        np.random.seed(self.seed)

        distribution_info = self.model.get_parameter_distribution_info()
        num_params = len(distribution_info)
        self.num_params = num_params

        # Number of sensitivity indices
        num_indices = 2**num_params - 1
        self.num_indices = num_indices


        # arrays to store our two samples X and X_tilde
        X = np.ones((self.num_samples, num_params))
        X_tilde = np.ones((self.num_samples, num_params))

        # array with samples
        self.samples = np.ones((self.num_samples, num_indices+1, num_params))

        i = 0
        for distribution in distribution_info:
            # get appropriate random number generator
            random_number_generator = getattr(np.random, distribution['distribution'])
            # make a copy
            my_args = list(distribution['distribution_parameter'])
            my_args.extend([self.num_samples])
            X[:, i] = random_number_generator(*my_args)
            X_tilde[:, i] = random_number_generator(*my_args)
            i += 1

        # making all possible combinations between the factors of X and X_tilde
        # loop to store all combinations with only one factor from X
        # First generate all indices of the combinations
        p = dict()
        ind = 0
        for k in range(num_params):
            for subset in combinations(range(num_params), k):
                p[ind] = np.asarray(subset)
                ind = ind+1
        del p[0]
        # Generate now the tensor of size (num_indices+1, )
        self.samples[:, 0, :] = X
        self.samples[:, num_indices, :] = X_tilde
        ps_temp = np.ones((self.num_samples, num_params))

        k = 1
        for i in range(num_indices-1):
            ps_temp[:, p[i+1]] = X[:, p[i+1]]
            ps_temp[:, p[num_indices-1-i]] = X_tilde[:, p[num_indices-1-i]]
            self.samples[:, k, :] = ps_temp
            k = k +1
        #print("Shape of self.samples {}".format(self.samples.shape))
        #exit()

    def get_all_samples(self):
        """ Return all samples

        Returns:
            np.array:    array with all combinations for all samples. The array
                         is of size (num_samples,num_indices+1,num_params),
                         it stores vertically the different possible combinations
                         to compute sensitivity indices.
        """
        #return np.reshape(self.samples, (-1, self.num_params), order='C')
        return self.samples

    def core_run(self):
        """ Run Analysis on model """
        # TODO find out why we have this dimension
        outputs = np.zeros((1, self.num_samples, self.num_indices+1))

        for s in range(1+(self.num_params*(self.num_params+1))//2):
            my_samples = self.get_all_samples()[:, s, :]
            self.model.update_model_from_sample_batch(my_samples)
            outputs[0, :, s] = np.reshape(self.eval_model(), (-1))
        for s in range(self.num_indices-self.num_params-1, self.num_indices+1):
            my_samples = self.get_all_samples()[:, s, :]
            self.model.update_model_from_sample_batch(my_samples)
            outputs[0, :, s] = np.reshape(self.eval_model(), (-1))

        S = self.__analyze(outputs)
        self.__print_results(S)

    def post_run(self):
        """ Analyze the results """
        pass

    def __analyze(self, Y):
        """ Compute sensitivity indices for given samples Y
        Y should have been computed with the SobolGratietDesigner, first the
        design set is generated with the Designer and then the function or
        metamodel is evaluated on it to give Y. When we evaluate a particular
        function on one design set Y is a matrix of size (num_samples, num_indices+1),
        and when we have several realizations of a Gaussian process we evaluate
        them on the design set X and then store them in a tensor of size
        (output_samples, num_samples, num_indices+1)

        Args:
        Y (numpy.array) :
            NumPy array containing the model outputs)
        Returns (dict) :
            dictionary with sensitivity indices
        """
        Y = np.asarray(Y)
        num_samples = len(Y[0, :, :])
        S = self.__create_Si_dict()
        num_indices = 2**self.num_params - 1

        for j in range(self.num_params):
            # First-order indices
            S_M_N_K_L = self.__compute_sensitivity_indice(Y[:, :, 0], Y[:, :, j+1], num_samples)
            S['S1'][j] = np.mean(S_M_N_K_L)
            S['S1_conf'][j] = self.__compute_confidence_interval(S_M_N_K_L)
            # Total order indices
            S_M_N_K_L_total = self.__compute_sensitivity_indice(Y[:, :, 0],
                                                                Y[:, :, num_indices-j-1],
                                                                num_samples)
            S['ST'][j] = 1-np.mean(S_M_N_K_L_total)
            S['ST_conf'][j] = self.__compute_confidence_interval(S_M_N_K_L_total)

        # Second order (+conf.)
        count = self.num_params + 1
        if self.calc_second_order:
            for j in range(self.num_params):
                for k in range(j + 1, self.num_params):
                    S_M_N_K_L_2 = self.__compute_sensitivity_indice(Y[:, :, 0],
                                                                    Y[:, :, count],
                                                                    num_samples)
                    S['S2'][j, k] = np.mean(S_M_N_K_L_2) - S['S1'][j] - S['S1'][k]
                    S['S2_conf'][j, k] = self.__compute_confidence_interval(S_M_N_K_L_2)
                    count = count +1
        return S

    def __generate_bootstrap_samples(self,num_samples):
        """ Generate the samples necessary to make bootstrapping
        Args :
        num_samples (int) :
            number of samples
        Returns:
            (np.array): array containing bootstrap permutations
        """
        # first row of the bootstrap indices matrix is unchanged, to fit to the
        # algorithm of Le Gratiet
        bootstrap_index = np.random.randint(num_samples, size = (self.num_bootstrap_samples,num_samples))
        bootstrap_index[0,:] = range(num_samples)
        return bootstrap_index

    def __compute_sensitivity_indice(self,Y,Y_tilde,num_samples):
        """ Compute the first-order sensitivity indices
        Args:
        Y (numpy.array):
            Evaluation of our sample X from the input space with the model (or
            metamodel) of our problem.
        Y_tilde (numpy.array) :
            Evaluation of our sample X_tilde from the input space with the model
            (or metamodel) of our problem.
        num_samples (int) :
            number of samples
        """
        # Estimator following Le Gratiet et al. 2014 Algorithm 1 and Proposition 1
        # init array
        S_M_N_K_L= np.zeros((self.output_samples, self.num_bootstrap_samples))
        H = self.__generate_bootstrap_samples(num_samples)
        # loop over realizations if there are any
        for k in range(self.output_samples):
            # loop to make bootstrapping over each realization
            for l in range(self.num_bootstrap_samples):
                Y_b = Y[k, H[l, :]]
                Y_tilde_b = Y_tilde[k, H[l, :]]
                numerator = (np.sum(Y_b*Y_tilde_b)/num_samples-(np.sum(Y_b+Y_tilde_b)/(2*num_samples))**2)
                denominator = (np.sum(Y_b**2)/num_samples-(np.sum(Y_b+Y_tilde_b)/(2*num_samples))**2)
                S_M_N_K_L[k, l] = numerator/denominator
        return S_M_N_K_L

    def __compute_confidence_interval(self, S_M_N_K_L):
        """Function to compute the confidence intervals
        Args:
        S_M_N_K_L (numpy.array) :
            NumPy array containing all the values of the Sensitivity Indices get
            thanks to Algorithm 1 and Proposition 1 in [1]
        Returns (float) :
            Value of the range of the confidence interval divided by 2. To have
            the exact confidence interval, compute [mean-result, mean+result]
        """
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        S_M_N_K_L_temp = S_M_N_K_L.reshape((1, self.num_bootstrap_samples*self.output_samples))
        conf_interval = norm.ppf(0.5 + self.confidence_level/2)*S_M_N_K_L_temp.std(ddof = 1)
        return conf_interval

    def __create_Si_dict(self):
        """ Create a dictionnary to store the results [SALib]"""
        S = dict((k, np.zeros(self.num_params)) for k in ('S1','S1_conf','ST','ST_conf'))
        if self.calc_second_order:
            S['S2'] = np.zeros((self.num_params, self.num_params))
            S['S2'][:] = np.nan
            S['S2_conf'] = np.zeros((self.num_params, self.num_params))
            S['S2_conf'][:] = np.nan
        return S

    def __print_results(self,S):
        """ Function to print results

        Args:

            S (dict): Dictionary with values of the sensitivity indices

        """
        parameter_names = self.model.get_parameter_names()
        title = 'Parameter'
        print('%s   S1       S1_conf    ST    ST_conf' % title)
        j = 0
        for name in parameter_names:
            print('%s %f %f %f %f' % (name + '       ', S['S1'][j], S['S1_conf'][j],
                                      S['ST'][j], S['ST_conf'][j]))
            j = j+1

        if self.calc_second_order:
            print('\n%s_1 %s_2    S2      S2_conf' % (title, title))
            for j in range(self.num_params):
                for k in range(j + 1, self.num_params):
                    print("%s %s %f %f" % (parameter_names[j] + '            ', parameter_names[k] + '      ',
                                           S['S2'][j, k], S['S2_conf'][j, k]))
