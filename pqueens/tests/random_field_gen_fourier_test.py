'''
Created on April 19th 2017
@author: jbi

'''

import unittest
import numpy as np
from scipy.stats import norm

from  pqueens.randomfields.random_field_gen_fourier_1d import RandomFieldGenFourier1D
from  pqueens.randomfields.univariate_field_generator_factory import UniVarRandomFieldGeneratorFactory

class Test(unittest.TestCase):

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

        self.my_field_generator = UniVarRandomFieldGeneratorFactory.create_new_random_field_generator(
            self.marginal_pdf,
            self.dimension,
            self.corrstruct,
            self.corr_length,
            self.energy_frac,
            self.field_bbox,
            self.num_terms_per_dim,
            self.total_terms)

    def test_constructor(self):
        my_stoch_dim = self.my_field_generator.get_stoch_dim()
        self.assertEqual(my_stoch_dim, 240,'Stochastic dimension is not correct!')
        loc = np.array([1,2.3,23])
        np.random.seed(self.seed)
        xi = np.random.randn(my_stoch_dim,1)
        my_vals = self.my_field_generator.evaluate_field_at_location(loc,xi)
        # last two arguments are relative and absolute tolerance, repectively
        np.testing.assert_allclose(my_vals,np.array([0.92106863, 0.96609547, 1.17947924]),1e-07,1e-07)

if __name__ == '__main__':
    unittest.main()
