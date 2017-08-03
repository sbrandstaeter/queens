from .abstract_designer import AbstractDesigner
import numpy as np
import math

class PseudoSaltelliDesigner(AbstractDesigner):
    """ Pseudo Saltelli designer for experiments

    References:

    [1] Le Gratiet, L., Cannamela, C., & Iooss, B. (2014).
        "A Bayesian Approach for Global Sensitivity Analysis
        of (Multifidelity) Computer Codes." SIAM/ASA Uncertainty
        Quantification Vol. 2. pp. 336-363,
        doi:10.1137/13926869

    Attributes:

        self.num_samples (int): number of design points
        self.ps (np.array): array with all samples/design points
        # TODO @Cecile add missing attributes here

    """
    def __init__(self,params,seed,num_samples):
        """
        Args:
            params (dict):
                Two samples X and X_tilde defining the input space,
                as in Algorithm 1 [1].
            seed (int):
                Seed for random number generation
            num_samples (int):
                Number of desired (random) samples


        """
        numparams = len(params)
        self.dim = len(params['X'])
        self.nb_combi = (self.dim+2+math.factorial(self.dim)//(2*math.factorial(self.dim-2)))
        print(self.nb_combi)
        # fix seed of random number generator
        np.random.seed(seed)
        self.num_samples = num_samples
        # array with samples
        self.ps = np.ones((self.nb_combi,num_samples,self.dim))

        self.X_Stock = np.ones((2,num_samples,self.dim))
        self.X = np.ones((num_samples,self.dim))
        self.X_tilde = np.ones((num_samples,self.dim))
        n = 0
        for key, value in params.items():
            i = 0
            if n < numparams:
                for k, v  in value.items():
                    self.X_Stock[n,:,i] = v
                    i = i + 1
                n = n + 1
                self.X = self.X_Stock[0,:,:]
                self.X_tilde = self.X_Stock[1,:,:]

    def get_all_samples(self):
        """ Generate all possible combinaisons from X and X_tilde
        necessary to compute the Sensitivity Indices with Algorithm 1 in [1]
        Returns:
            (np.array): array with all samples
        """

        for i in range(self.dim):
            X_tilde1 = self.X_tilde.copy()
            X_tilde1[:,i] = self.X[:,i]
            self.ps[i+1,:,:] = X_tilde1
            for j in range(i+1,self.dim):
                X_tilde2 = self.X_tilde.copy()
                X_tilde2[:,i] = self.X[:,i]
                X_tilde2[:,j] = self.X[:,j]
                self.ps[i+self.dim+j,:,:] = X_tilde2
        self.ps[0,:,:] = self.X
        self.ps[self.nb_combi-1,:,:] = self.X_tilde

        return self.ps
