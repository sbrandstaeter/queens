import numpy as np
import math
import os
import combi
import pyDOE
import random
from scipy.stats import norm

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
        Calculate second-order sensitivities (default True)
    num_bootstrap_samples (int) :
        The number of bootstrap samples (default 100)
    confidence_level (float) :
        The confidence interval level (default 0.95)
    output_samples (int):
        The number of output samples Y (default 1)
    """

    def __init__(self,params,calc_second_order, num_bootstrap_samples,
                confidence_level, output_samples, num_bootstrap_conf):
        """
        Args:
        params (dict):
            Two samples X and X_tilde defining the input space,
            as in Algorithm 1 [1].
        calc_second_order (bool) :
            Calculate second-order sensitivities (default True)
        num_bootstrap_samples (int) :
            The number of bootstrap samples (default 100)
        confidence_level (float) :
            The confidence interval level (default 0.95)
        output_samples (int):
            The number of output samples Y (default 1)
        num_bootstrap_conf (int) :
            Number of bootstrap iterations for the computation of confidence intervals
        """
        self.params = params
        self.dim = len(self.params['X'])
        self.calc_second_order = calc_second_order
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level  = confidence_level
        self.output_samples = output_samples
        self.num_bootstrap_conf = num_bootstrap_conf

    def generate_bootstrap_samples(self,m):
        """ Generate the samples necessary to make bootstrapping
        Args :
        m (int) :
            number of samples
        """
        H = np.zeros((self.num_bootstrap_samples,m), dtype = int)
        for z in range(self.num_bootstrap_samples):
            a = 1
            b = m
            liste =[]
            for j in range(m):
                liste.append(random.randint(a,b-1))
            H[z,:] = liste
            H[z,:] = np.round(H[z,:])
        return H

    def compute_first_order_sensitivity_indice(self,Y,Y_tilde,m):
        """ Compute the First-Order Indice Sensitivity Indices with
        Algorithm 1 and Proposition 1 in [1]
        Args:
        Y (numpy.array) :
            Evaluation of our sample X from the input space with the model (or
            metamodel) of our problem.
        Y_tilde (numpy.array) :
            Evaluation of our sample X_tilde from the input space with the model
            (or metamodel) of our problem.
        m (int) :
            number of samples
        """
        # First order estimator following Le Gratiet et al. 2014
        S_M_N_K_L= np.zeros((self.output_samples,self.num_bootstrap_samples),dtype=float)
        H = self.generate_bootstrap_samples(m)
        for k in range(self.output_samples):
            num = (np.sum(Y[k,:]*Y_tilde[k,:])/m-(np.sum(Y[k,:]+Y_tilde[k,:])/(2*m))**2)
            den = (sum(Y[k,:]**2)/m-(np.sum(Y[k,:]+Y_tilde[k,:])/(2*m))**2)
            S_M_N_K_L[k,0] = num/den
            for l in range(1,self.num_bootstrap_samples):
                Y_b = Y[k,H[l,:]]
                Y_tilde_b = Y_tilde[k,H[l,:]]
                num_b = (np.sum(Y_b*Y_tilde_b)/m-(np.sum(Y_b+Y_tilde_b)/(2*m))**2)
                den_b = (sum(Y_b**2)/m-(np.sum(Y_b+Y_tilde_b)/(2*m))**2)
                S_M_N_K_L[k,l] = num_b/den_b
        return S_M_N_K_L

    def compute_second_order_sensitivity_indice(self,Y, Y_tilde, Y_tilde_bis, Y_tilde_ter,m):
        """ Compute the Second-Order Indice Sensitivity Indices with
        Algorithm 1 and Proposition 1 in [1]
        Args:
        Y (numpy.array) :
            Evaluation of our sample X from the input space with the model (or
            metamodel) of our problem.
        Y_tilde (numpy.array) :
            Evaluation of our sample X_tilde from the input space with the model
            (or metamodel) of our problem.
        """
        S_M_N_K_L =  np.zeros((self.output_samples,self.num_bootstrap_samples), dtype = float)
        H = self.generate_bootstrap_samples(m)
        for k in range(self.output_samples):
            num = (np.sum(Y[k,:]*Y_tilde[k,:])/m-(np.sum(Y[k,:]+Y_tilde[k,:])/(2*m))**2)
            den = (sum(Y[k,:]**2)/m-(np.sum(Y[k,:]+Y_tilde[k,:])/(2*m))**2)
            S_1 = np.mean(self.compute_first_order_sensitivity_indice(Y, Y_tilde_bis,m))
            S_2 = np.mean(self.compute_first_order_sensitivity_indice(Y, Y_tilde_ter,m))
            S_M_N_K_L[k,0] = num/den - S_1 - S_2
            for l in range(1,self.num_bootstrap_samples):
                Y_b = Y[:,H[l,:]]
                Y_tilde_b = Y_tilde[:,H[l,:]]
                Y_tilde_bis_b = Y_tilde_bis[:,H[l,:]]
                Y_tilde_ter_b = Y_tilde_ter[:,H[l,:]]
                num_b = (np.sum(Y_b[k,:]*Y_tilde_b[k,:])/m-(np.sum(Y_b[k,:]+Y_tilde_b[k,:])/(2*m))**2)
                den_b = (sum(Y_b[k,:]**2)/m-(np.sum(Y_b[k,:]+Y_tilde_b[k,:])/(2*m))**2)
                S_1_b = np.mean(self.compute_first_order_sensitivity_indice(Y_b, Y_tilde_bis_b,m))
                S_2_b = np.mean(self.compute_first_order_sensitivity_indice(Y_b, Y_tilde_ter_b,m))
                S_M_N_K_L[k,l] = num_b/den_b - S_1_b - S_2_b
        return S_M_N_K_L

    def compute_total_order_sensitivity_indice(self,Y, Y_tilde,m):
        """ Compute the Total-Order Indice Sensitivity Indices [2]
        Args:
        Y (numpy.array) :
            Evaluation of our sample X from the input space with the model (or
            metamodel) of our problem.
        Y_tilde (numpy.array) :
            Evaluation of our sample X_tilde from the input space with the model
            (or metamodel) of our problem.
        m (int) :
            number of samples
        """
        S_M_N_K_L = np.zeros((self.output_samples,self.num_bootstrap_samples),dtype=float)
        H = self.generate_bootstrap_samples(m)
        for k in range(self.output_samples):
            num = (np.sum(Y[k,:]*Y_tilde[k,:])/m-(np.sum(Y[k,:]+Y_tilde[k,:])/(2*m))**2)
            den = (sum(Y[k,:]**2)/m-(np.sum(Y[k,:]+Y_tilde[k,:])/(2*m))**2)
            S_M_N_K_L[k,0] = 1-num/den
            for l in range(1,self.num_bootstrap_samples):
                Y_b = Y[k,H[l,:]]
                Y_tilde_b = Y_tilde[k,H[l,:]]
                num_b = (np.sum(Y_b*Y_tilde_b)/m-(np.sum(Y_b+Y_tilde_b)/(2*m))**2)
                den_b = (sum(Y_b**2)/m-(np.sum(Y_b+Y_tilde_b)/(2*m))**2)
                S_M_N_K_L[k,l] = 1-num_b/den_b
        return S_M_N_K_L

    def compute_confidence_interval(self, S_M_N_K_L):
        """Function to compute the confidence intervals for our sensitivity
        indice
        Args:
        S_M_N_K_L (numpy.array) :
            NumPy array containing all the values of the Sensitivity Indices get
            thanks to Algorithm 1 and Proposition 1 in [1]
        """
        S_M_N_K_L_bootstrap = np.zeros([self.num_bootstrap_samples*self.output_samples])
        data_bootstrap = np.zeros([self.num_bootstrap_conf])
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        S_M_N_K_L_temp = S_M_N_K_L.reshape((1,self.num_bootstrap_samples*self.output_samples))
        bootstrap_index = np.random.randint(len(S_M_N_K_L_temp), size = (self.num_bootstrap_samples*self.output_samples,self.num_bootstrap_conf))
        S_M_N_K_L_bootstrap= S_M_N_K_L[bootstrap_index]
        data_bootstrap = np.average(np.abs(S_M_N_K_L_bootstrap), axis = 1)
        return norm.ppf(0.5 + self.confidence_level/2)*data_bootstrap.std(ddof = 1)

    def create_Si_dict(self):
        """ Create a dictionnary to store the results, function from
        the python library SALib """
        S = dict((k,np.zeros(self.dim)) for k in ('S1','S1_conf','ST','ST_conf'))
        if self.calc_second_order:
            S['S2'] = np.zeros((self.dim, self.dim))
            S['S2'][:] = np.nan
            S['S2_conf'] = np.zeros((self.dim, self.dim))
            S['S2_conf'][:] = np.nan
        return S

    def analyze(self,Y):
        """ Compute sensitivity indices for given samples Y
        Args:
        Y (numpy.array) :
            NumPy array containing the model outputs)
        Returns (dict) :
            dictionary with sensitivity indices
        """
        Y = np.asarray(Y)
        m = len(Y[0,:,:])
        S = self.create_Si_dict()
        nb_combi = (self.dim+2+math.factorial(self.dim)//(2*math.factorial(self.dim-2)))

        for j in range(self.dim):
            S_M_N_K_L = self.compute_first_order_sensitivity_indice(Y[:,:,0], Y[:,:,j+1],m)
            S['S1'][j] = np.mean(S_M_N_K_L)
            S['S1_conf'][j] = self.compute_confidence_interval(S_M_N_K_L)
            S_M_N_K_L_total = self.compute_total_order_sensitivity_indice(Y[:,:,0], Y[:,:,nb_combi-j-2],m)
            S['ST'][j] = np.mean(S_M_N_K_L_total)
            S['ST_conf'][j] = self.compute_confidence_interval(S_M_N_K_L_total)

        # Second order (+conf.)
        if self.calc_second_order:
            for j in range(self.dim):
                for k in range(j + 1, self.dim):
                    S_M_N_K_L_2 = self.compute_second_order_sensitivity_indice(Y[:,:,0],
                    Y[:,:,j+k+self.dim], Y[:,:,j+1], Y[:,:,k+1],m)
                    S['S2'][j, k] = np.mean(S_M_N_K_L_2)
                    S['S2_conf'][j, k] = self.compute_confidence_interval(S_M_N_K_L_2)

        print("{0:<30} {1:>10} {2:>10} {3:>15} {4:>10}".format(
            "Parameter",
            "S1",
            "S1_conf",
            "ST",
            "ST_conf")
        )
        for j in list(range(self.dim)):
            print("{0!s:30} {1!s:10} {2!s:10} {3!s:15} {4!s:10}".format(
                print('x',j),
                S['S1'][j],
                S['S1_conf'][j],
                S['ST'][j],
                S['ST_conf'][j]))
