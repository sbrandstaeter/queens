import numpy as np
import math
import random
from scipy.stats import norm
from scipy import stats

class SobolAnalyzer(object):
    """ Class for computing Sobol indices

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

    problem  (dict) :
        The problem definition
    calc_second_order (bool) :
        Calculate second-order sensitivities
    num_bootstrap_samples (int) :
        The number of bootstrap samples
    confidence_level (float) :
        The confidence interval level
    output_samples (int):
        The number of output samples Y
    """

    def __init__(self,params,calc_second_order , num_bootstrap_samples ,
                confidence_level , output_samples):
        """
        Args:
        params (dict):

        calc_second_order (bool) :
            Calculate second-order sensitivities
        num_bootstrap_samples (int) :
            The number of bootstrap samples
        confidence_level (float) :
            The confidence interval level
        output_samples (int):
            The number of output samples Y
        """
        self.params = params
        self.numparams = len(self.params)
        self.calc_second_order = calc_second_order
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level  = confidence_level
        self.output_samples = output_samples

    def analyze(self,Y):
        """ Compute sensitivity indices for given samples Y
        Y should have been computed with the SobolGratietDesigner, first the
        design set is generated with the Designer and then the function or
        metamodel is evaluated on it to give Y. When we evaluate a particular
        function on one design set Y is a matrix of size (num_samples, nb_indices+1),
        and when we have several realizations of a Gaussian process we evaluate
        them on the design set X and then store them in a tensor of size
        (output_samples, num_samples, nb_indices+1)

        Args:
        Y (numpy.array) :
            NumPy array containing the model outputs)
        Returns (dict) :
            dictionary with sensitivity indices
        """
        Y = np.asarray(Y)
        num_samples = len(Y[0,:,:])
        S = self.create_Si_dict()
        nb_indices = 2**self.numparams - 1

        for j in range(self.numparams):
            # First-order indices
            S_M_N_K_L = self.compute_sensitivity_indice(Y[:,:,0], Y[:,:,j+1],num_samples)
            S['S1'][j] = np.mean(S_M_N_K_L)
            S['S1_conf'][j] = self.compute_confidence_interval(S_M_N_K_L)
            # Total order indices
            S_M_N_K_L_total = self.compute_sensitivity_indice(Y[:,:,0], Y[:,:,nb_indices-j-1],num_samples)
            S['ST'][j] = 1-np.mean(S_M_N_K_L_total)
            S['ST_conf'][j] = self.compute_confidence_interval(S_M_N_K_L_total)

        # Second order (+conf.)
        count = self.numparams + 1
        if self.calc_second_order:
            for j in range(self.numparams):
                for k in range(j + 1, self.numparams):
                    S_M_N_K_L_2 = self.compute_sensitivity_indice(Y[:,:,0],
                    Y[:,:,count],num_samples)
                    S['S2'][j, k] = np.mean(S_M_N_K_L_2) - S['S1'][j] - S['S1'][k]
                    S['S2_conf'][j, k] = self.compute_confidence_interval(S_M_N_K_L_2)
                    count = count +1
        return S

    def generate_bootstrap_samples(self,num_samples):
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

    def compute_sensitivity_indice(self,Y,Y_tilde,num_samples):
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
        S_M_N_K_L= np.zeros((self.output_samples,self.num_bootstrap_samples))
        H = self.generate_bootstrap_samples(num_samples)
        # loop over realizations if there are any
        for k in range(self.output_samples):
            # loop to make bootstrapping over each realization
            for l in range(self.num_bootstrap_samples):
                Y_b = Y[k,H[l,:]]
                Y_tilde_b = Y_tilde[k,H[l,:]]
                numerator = (np.sum(Y_b*Y_tilde_b)/num_samples-(np.sum(Y_b+Y_tilde_b)/(2*num_samples))**2)
                denominator = (np.sum(Y_b**2)/num_samples-(np.sum(Y_b+Y_tilde_b)/(2*num_samples))**2)
                S_M_N_K_L[k,l] = numerator/denominator
        return S_M_N_K_L

    def compute_confidence_interval(self, S_M_N_K_L):
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
        S_M_N_K_L_temp = S_M_N_K_L.reshape((1,self.num_bootstrap_samples*self.output_samples))
        conf_interval = norm.ppf(0.5 + self.confidence_level/2)*S_M_N_K_L_temp.std(ddof = 1)
        return conf_interval

    def create_Si_dict(self):
        """ Create a dictionnary to store the results [SALib]"""
        S = dict((k,np.zeros(self.numparams)) for k in ('S1','S1_conf','ST','ST_conf'))
        if self.calc_second_order:
            S['S2'] = np.zeros((self.numparams,self.numparams))
            S['S2'][:] = np.nan
            S['S2_conf'] = np.zeros((self.numparams, self.numparams))
            S['S2_conf'][:] = np.nan
        return S

    def print_results(self,S):
        """ Function to print the results inspired from [SALib]
        Args:
        S (dict): dictionnary with all values of the sensitivity indices
        """
        title = 'Parameter'
        print('%s   S1       S1_conf    ST    ST_conf' % title)
        j = 0
        for name in self.params.keys():
            print('%s %f %f %f %f' % (name + '       ', S['S1'][j], S['S1_conf'][j],
            S['ST'][j], S['ST_conf'][j]))
            j = j+1

        if self.calc_second_order == True:
            print('\n%s_1 %s_2    S2      S2_conf' % (title,title))
            j = 0
            params_temp = self.params.copy()
            for name in self.params.keys():
                del params_temp[name]
                k = j+1
                for name_b in params_temp.keys():
                    if k < self.numparams:
                        print("%s %s %f %f" % (name + '            ', name_b + '      ', S['S2'][j, k],
                        S['S2_conf'][j, k]))
                        k = k+1
                j = j+1
