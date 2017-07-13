import numpy as np
import math
import os
import combi
import pyDOE
import random
from pqueens.sensitivity_analyzers.generate_sample import fact

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
    num_samples = 1000
    def __init__(self,params = {'X':{'x1' : np.random.uniform(0,1,num_samples), 
            'x2' : np.random.uniform(0,1,num_samples)},
            'X_tilde' :{'x1' : np.random.uniform(0,1,num_samples), 
            'x2' : np.random.uniform(0,1,num_samples)}},
                calc_second_order = True , num_bootstrap_samples = 100, 
                confidence_level = 0.95, output_samples = 1)
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
        """
        self.params = params
        self.calc_second_order = calc_second_order
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level  = confidence_level
        self.output_samples = output_samples

    def generate_bootstrap_samples(self,m):
        """ Generate the samples necessary to make bootstrapping """
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

    def compute_first_order_sensitivity_indice(self,Y,Y_tilde):
        """ Compute the First-Order Indice Sensitivity Indices with 
        Algorithm 1 and Proposition 1 in [1] """
        # First order estimator following Le Gratiet et al. 2014
        S_M_N_K_L= np.zeros((self.output_samples,self.num_bootstrap_samples),dtype=float)
        H = self.bootstrap(len(Y))
        m = len(Y)
        for k in range(self.output_samples):
            num = (np.sum(Y*Y_tilde)/m-(np.sum(Y+Y_tilde)/(2*m))**2)
            den = (sum(Y**2)/m-(np.sum(Y+Y_tilde)/(2*m))**2)
            S_M_N_K_L[k,0] = num/den
            for l in range(1,self.num_bootstrap_samples):
                Y_b = Y[H[l,:]]
                Y_tilde_b = Y_tilde[H[l,:]]
                num_b = (np.sum(Y_b*Y_tilde_b)/m-(np.sum(Y_b+Y_tilde_b)/(2*m))**2)
                den_b = (sum(Y_b**2)/m-(np.sum(Y_b+Y_tilde_b)/(2*m))**2)
                S_M_N_K_L[k,l] = num_b/den_b
        return np.mean(S_M_N_K_L)

    def compute_second_order_sensitivity_indice(self,Y, Y_tilde, Y_tilde_bis, Y_tilde_ter):
        """ Compute the Second-Order Indice Sensitivity Indices with 
        Algorithm 1 and Proposition 1 in [1] """
        S_M_N_K_L =  np.zeros((self.output_samples,self.num_bootstrap_samples), dtype = float)
        H = self.bootstrap(len(Y))
        m = len(Y)
        for k in range(self.output_samples):
            num = (np.sum(Y*Y_tilde)/m-(np.sum(Y+Y_tilde)/(2*m))**2)
            den = (sum(Y**2)/m-(np.sum(Y+Y_tilde)/(2*m))**2)
            S_1 = self.first_order_indice(Y, Y_tilde_bis)
            S_2 = self.first_order_indice(Y, Y_tilde_ter) 
            S_M_N_K_L[k,0] = num/den - S_1 - S_2
            for l in range(1,self.num_bootstrap_samples):
                Y_b = Y[H[l,:]]
                Y_tilde_b = Y_tilde[H[l,:]]
                Y_tilde_bis_b = Y_tilde_bis[H[l,:]]
                Y_tilde_ter_b = Y_tilde_ter[H[l,:]]
                num_b = (np.sum(Y_b*Y_tilde_b)/m-(np.sum(Y_b+Y_tilde_b)/(2*m))**2)
                den_b = (sum(Y_b**2)/m-(np.sum(Y_b+Y_tilde_b)/(2*m))**2)
                S_1_b = self.first_order_indice(Y_b, Y_tilde_bis_b)
                S_2_b = self.first_order_indice(Y_b, Y_tilde_ter_b) 
                S_M_N_K_L[k,l] = num_b/den_b - S_1_b - S_2_b
        return np.mean(S_M_N_K_L)

    def compute_total_order_sensitivity_indice(self,Y, Y_tilde):
        """ Compute the Total-Order Indice Sensitivity Indices [2] """
        S_M_N_K_L = np.zeros((self.output_samples,self.num_bootstrap_samples),dtype=float)
        H = self.bootstrap(len(Y))
        m = len(Y)
        for k in range(self.output_samples):
            num = (np.sum(Y*Y_tilde)/m-(np.sum(Y+Y_tilde)/(2*m))**2)
            den = (sum(Y**2)/m-(np.sum(Y+Y_tilde)/(2*m))**2)
            S_M_N_K_L[k,0] = 1-num/den
            for l in range(1,self.num_bootstrap_samples):
                Y_b = Y[H[l,:]]
                Y_tilde_b = Y_tilde[H[l,:]]
                num_b = (np.sum(Y_b*Y_tilde_b)/m-(np.sum(Y_b+Y_tilde_b)/(2*m))**2)
                den_b = (sum(Y_b**2)/m-(np.sum(Y_b+Y_tilde_b)/(2*m))**2)
                S_M_N_K_L[k,l] = 1-num_b/den_b
        return np.mean(S_M_N_K_L)

    def create_Si_dict(self,dim):
        """ Create a dictionnary to store the results, function from
        the python library SALib """
        S = dict((k,np.zeros(dim)) for k in ('S1','S1_conf','ST','ST_conf'))
        if self.calc_second_order:
            S['S2'] = np.zeros((dim, dim))
            S['S2'][:] = np.nan
            S['S2_conf'] = np.zeros((dim, dim))
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
        # determining the dimension of the problem 
        dim = len(self.params['X'])
        Y = np.asarray(Y)
        m = len(Y)
        S = self.create_Si_dict(dim)
        nb_combi = (dim+2+fact(dim)//(2*fact(dim-2)))

        for j in range(dim):
            S['S1'][j] = self.first_order_indice(Y[:,0], Y[:,j+1])
            S['S1_conf'][j] = 0
            S['ST'][j] = self.total_order_indice(Y[:,0], Y[:,nb_combi-j-2])
            S['ST_conf'][j] = 0

        # Second order (+conf.)
        if self.calc_second_order:
            for j in range(dim):
                for k in range(j + 1, dim):
                    S['S2'][j, k] = self.second_order_indice(Y[:,0], 
                    Y[:,j+k+dim], Y[:,j+1], Y[:,k+1])
                    S['S2_conf'][j, k] =  0
        return S
