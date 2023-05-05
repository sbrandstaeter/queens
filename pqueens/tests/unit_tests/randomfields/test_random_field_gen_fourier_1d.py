"""Created on April 19th 2017.

@author: jbi
"""

import unittest

import numpy as np
from scipy import stats
from scipy.stats import norm

from pqueens.randomfields.univariate_field_generator_factory import (
    UniVarRandomFieldGeneratorFactory,
)


class TestRandomFieldGeneratorFourier1D(unittest.TestCase):
    """TODO_doc."""

    def setUp(self):
        """TODO_doc."""
        # setup some necessary variables to setup random field generators
        self.dimension = 1
        self.corrstruct = 'squared_exp'
        self.corr_length = 25
        self.energy_frac = 0.95
        self.field_bbox = np.array([-100, 100])
        self.marginal_pdf = norm(0, 1)
        self.num_terms_per_dim = 120
        self.total_terms = 120
        self.loc = [0, 10, 25, 100]
        self.seed = 42
        self.my_field_generator = (
            UniVarRandomFieldGeneratorFactory.create_new_random_field_generator(
                marg_pdf=self.marginal_pdf,
                spatial_dimension=self.dimension,
                corrstruct=self.corrstruct,
                corr_length=self.corr_length,
                energy_frac=self.energy_frac,
                field_bbox=self.field_bbox,
                num_terms_per_dim=self.num_terms_per_dim,
                total_terms=self.total_terms,
            )
        )

        self.my_stoch_dim = self.my_field_generator.get_stoch_dim()

    # should trigger error because desired energy fraction not reached
    def test_not_enough_fourier_terms(self):
        """TODO_doc."""
        with self.assertRaises(RuntimeError):
            UniVarRandomFieldGeneratorFactory.create_new_random_field_generator(
                marg_pdf=self.marginal_pdf,
                spatial_dimension=self.dimension,
                corrstruct=self.corrstruct,
                corr_length=self.corr_length,
                energy_frac=0.9999,
                field_bbox=self.field_bbox,
                num_terms_per_dim=5,
                total_terms=5,
            )

    # should trigger error because number of phase angles do not match stochastic
    # dimension
    def test_wrong_number_phase_angles(self):
        """TODO_doc."""
        with self.assertRaises(RuntimeError):
            mystuff = UniVarRandomFieldGeneratorFactory.create_new_random_field_generator(
                marg_pdf=self.marginal_pdf,
                spatial_dimension=self.dimension,
                corrstruct=self.corrstruct,
                corr_length=self.corr_length,
                energy_frac=self.energy_frac,
                field_bbox=self.field_bbox,
                num_terms_per_dim=self.num_terms_per_dim,
                total_terms=self.total_terms,
            )
            mystuff.gen_sample_gauss_field(10, np.array((4, 4)))

    def test_values_at_location(self):
        """TODO_doc."""
        loc = np.array([0, 25, 50, 100])
        np.random.seed(self.seed)
        xi = np.random.randn(self.my_stoch_dim, 1)
        my_vals = self.my_field_generator.evaluate_field_at_location(loc, xi)

        # last two arguments are relative and absolute tolerance, respectively
        np.testing.assert_allclose(
            my_vals, np.array([0.88594784, 1.08872132, 0.88706044, -0.81812818]), 1e-07, 1e-07
        )

    def test_correlation(self):
        """TODO_doc."""
        loc = np.array([0, 10, 25, 100])
        my_vals = np.zeros((loc.shape[0], 1000))
        np.random.seed(self.seed)
        for i in range(200):
            xi = np.random.randn(self.my_stoch_dim, 1)
            my_vals[:, i] = self.my_field_generator.evaluate_field_at_location(loc, xi)

        # compute empirical correlation coefficient
        act_corr_at_dist_10 = np.corrcoef(my_vals[0, :], my_vals[1, :])
        act_corr_at_dist_25 = np.corrcoef(my_vals[0, :], my_vals[2, :])
        act_corr_at_dist_100 = np.corrcoef(my_vals[0, :], my_vals[3, :])

        # expected correlation
        # exp_corr_at_dist_10     = exp(-(loc(2)-loc(1))^2/corr_length^2);
        # exp_corr_at_dist_25     = exp(-(loc(3)-loc(1))^2/corr_length^2);
        # exp_corr_at_dist_100    = exp(-(loc(4)-loc(1))^2/corr_length^2);

        ref_corr_at_dist_10 = 0.834671507078
        ref_corr_at_dist_25 = 0.375379129134
        # becomes positive if more samples were used
        ref_corr_at_dist_100 = -0.0197084550257

        self.assertAlmostEqual(
            act_corr_at_dist_10[0, 1],
            ref_corr_at_dist_10,
            7,
            'Correlation for distance 10 is not correct.',
        )
        self.assertAlmostEqual(
            act_corr_at_dist_25[0, 1],
            ref_corr_at_dist_25,
            7,
            'Correlation for distance 25 is not correct.',
        )
        self.assertAlmostEqual(
            act_corr_at_dist_100[0, 1],
            ref_corr_at_dist_100,
            7,
            'Correlation for distance 100 is not correct.',
        )

    def test_marginal_distribution(self):
        """TODO_doc."""
        my_vals = np.zeros((1, 200))
        np.random.seed(self.seed)
        for i in range(200):
            xi = np.random.randn(self.my_stoch_dim, 1)
            my_vals[0, i] = self.my_field_generator.evaluate_field_at_location(np.array([30.0]), xi)

        # try to check whether marginal distribution is normally distributed
        # using kstest
        test_statistic = (stats.kstest(my_vals[0, :], 'norm'))[0]
        self.assertAlmostEqual(test_statistic, 0.030059684522705243)


if __name__ == '__main__':
    unittest.main()
