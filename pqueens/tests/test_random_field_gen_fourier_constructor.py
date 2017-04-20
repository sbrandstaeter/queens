'''
Created on April 19th 2017
@author: jbi

'''

import unittest
import numpy as np
from scipy.stats import norm

from  pqueens.randomfields.random_field_gen_fourier_1d import RandomFieldGenFourier1D
from  pqueens.randomfields.univariate_field_generator_factory import UniVarRandomFieldGeneratorFactory

class TestRandomFieldGeneratorConstruction(unittest.TestCase):

    def setUp(self):
        # setup some necessary variables to setup random field generators
        self.dimension               = 1
        self.corrstruct              = 'squared_exp'
        self.corr_length             = 25
        self.energy_frac             = 0.95
        self.field_bbox              = np.array([-100, 100])
        self.marginal_pdf            = norm(0,1)
        self.num_terms_per_dim       = 120
        self.total_terms             = 120
        self.loc                     = [0, 10, 25, 100]
        self.seed                    = 42


    def test_construction_1d(self):

        my_field_generator = UniVarRandomFieldGeneratorFactory.create_new_random_field_generator(
            self.marginal_pdf,
            self.dimension,
            self.corrstruct,
            self.corr_length,
            self.energy_frac,
            self.field_bbox,
            self.num_terms_per_dim,
            self.total_terms)

        my_stoch_dim = my_field_generator.get_stoch_dim()
        self.assertEqual(my_stoch_dim, 240,'Stochastic dimension is not correct!')

        # trigger error because desired energy fraction not reached
        # trigger error because corrlength to large w.r.t. bounding box

        #def test_construction(self):

        #self.assertEqual(self.my_stoch_dim, 240,'Stochastic dimension is not correct!')

        # trigger error because desired energy fraction not reached
        # trigger error because corrlength to large w.r.t. bounding box


if __name__ == '__main__':
    unittest.main()
