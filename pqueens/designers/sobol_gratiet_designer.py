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
        self.ps (np.array): array with all combinations from our samples/design points
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
        nb_combi = (numparams+2+math.factorial(numparams)//(2*math.factorial(numparams-2)))
        # fix seed of random number generator
        np.random.seed(seed)
        self.num_samples = num_samples
        # arrays to store our two samples X and X_tilde
        X = np.ones((self.num_samples,numparams))
        X_tilde = np.ones((self.num_samples,numparams))
        # array with samples
        self.ps = np.ones((num_samples,nb_combi,numparams))

        i=0
        for _ ,value in params.items():
            # get appropriate random number generator
            random_number_generator = getattr(np.random, value['distribution'])
            # make a copy
            my_args = list(value['distribution_parameter'])
            my_args.extend([num_samples])
            X[:,i] = random_number_generator(*my_args)
            X_tilde[:,i] = random_number_generator(*my_args)
            i+=1

        # making all possible combinations between the factors of X and X_tilde
        # loop to store all combinations with only one factor from X, for first-order
        # sensitivity indices
        for i in range(numparams):
            X_tilde1 = X_tilde.copy()
            X_tilde1[:,i] = X[:,i]
            self.ps[:,i+1,:] = X_tilde1
        # loop to store all combinations with two factors from X, for second-order
        # sensitivity indices
            for j in range(i+1,numparams):
                X_tilde2 = X_tilde.copy()
                X_tilde2[:,i] = X[:,i]
                X_tilde2[:,j] = X[:,j]
                self.ps[:,i+numparams+j,:] = X_tilde2
        self.ps[:,0,:] = X
        self.ps[:,nb_combi-1,:] = X_tilde

    def sample_generator(self):
        """ Generator to iterate over experimental design """
        i = 0
        while i < self.num_samples:
            yield self.ps[i,:]
            i += 1

    def get_all_samples(self):
        """
        Returns:
            ps (np.array): array with all combinations for all samples
        """
        return self.ps
