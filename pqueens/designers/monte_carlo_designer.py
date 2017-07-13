from .abstract_designer import AbstractDesigner
import numpy as np

class MonteCarloDesigner(AbstractDesigner):
    """ Monte Carlo based design of experiments

    Attributes:
        self.num_samples (int): number of design points
        self.mc (np.array):    array with all samples/design points

    """

    def __init__(self,params,seed,num_samples):
        """
        Args:
            params (dict):
            seed (int) : Seed for random number generation
            num_samples (int) : Number of desired (random) samples

        """
        numparams = len(params)
        # fix seed of random number generator
        np.random.seed(seed)
        self.num_samples = num_samples
        self.mc = np.zeros((num_samples,numparams))

        i=0
        for _ ,value in params.items():
            # get appropriate random number generator
            random_number_generator = getattr(np.random, value['distribution'])
            # make a copy
            my_args = list(value['distribution_parameter'])
            my_args.extend([num_samples])
            self.mc[:,i] = random_number_generator(*my_args)
            i+=1


    def sample_generator(self):
        """ Generator to iterate over experimental design """
        i = 0
        while i < self.num_samples:
            yield self.mc[i,:]
            i += 1

    def get_all_samples(self):
        return self.mc
