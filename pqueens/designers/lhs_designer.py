from .abstract_designer import AbstractDesigner
from pyDOE import lhs
import numpy as np

class LatinHyperCubeDesigner(AbstractDesigner):
    """ Latin Hyper Cube designer for experiments

    Attributes:
        self.num_samples (int): number of design points
        self.lhd (np.array):    array with all samples/design points

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
        self.lhd = lhs(numparams, num_samples, 'maximin', iterations=10)
        i = 0
        for _ , value in params.items():
            self.lhd[:,i] = self.lhd[:,i]*(value['max']-value['min'])+value['min']
            i+=1


    def suggest_next_evaluation(self):
        """ Generator to iterate over experimental design """
        i = 0
        while i < self.num_samples:
            yield self.lhd[i,:]
            i += 1
