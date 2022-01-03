"""Created on April 21th 2017.

@author: jbi
"""

import unittest

import numpy as np
import pytest
from scipy import stats
from scipy.stats import norm

from pqueens.randomfields.random_field_gen_fourier_2d import RandomFieldGenFourier2D
from pqueens.randomfields.univariate_field_generator_factory import (
    UniVarRandomFieldGeneratorFactory,
)


class TestRandomFieldGeneratorFourier2D(unittest.TestCase):
    def setUp(self):
        # setup some necessary variables to setup random field generators
        self.dimension = 2
        self.corrstruct = 'squared_exp'
        self.corr_length = 25
        self.energy_frac = 0.95
        self.field_bbox = np.array([-100, 100, -100, 100])
        self.marginal_pdf = norm(0, 1)
        self.num_terms_per_dim = 10
        self.total_terms = 20
        self.loc = np.array([[0, 0], [0, 10], [10, 0], [0, 25], [25, 0], [0, 100], [100, 0]])
        self.seed = 42

        # pylint: disable=line-too-long
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
        # pylint: enable=line-too-long

        self.my_stoch_dim = self.my_field_generator.get_stoch_dim()

    # should trigger error because desired energy fraction not reached
    @pytest.mark.unit_tests
    def test_not_enough_fourier_terms(self):
        with self.assertRaises(RuntimeError):
            UniVarRandomFieldGeneratorFactory.create_new_random_field_generator(
                marg_pdf=self.marginal_pdf,
                spatial_dimension=self.dimension,
                corrstruct=self.corrstruct,
                corr_length=self.corr_length,
                energy_frac=0.99,
                field_bbox=self.field_bbox,
                num_terms_per_dim=10,
                total_terms=20,
            )

    # should trigger error because number of phase angles do not match stochastic
    # dimension
    @pytest.mark.unit_tests
    def test_wrong_number_phase_angles(self):
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
            mystuff.gen_sample_gauss_field(np.array([[10, 10]]), np.array((4, 4)))

    # should trigger error because dimension of location is of
    @pytest.mark.unit_tests
    def test_wrong_locatio_dimension(self):
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
            xi = np.random.randn(mystuff.get_stoch_dim(), 1)
            mystuff.gen_sample_gauss_field(np.array([[10, 10, 10]]), xi)

    @pytest.mark.unit_tests
    def test_values_at_location(self):
        np.random.seed(self.seed)
        xi = np.random.randn(self.my_stoch_dim, 1)
        my_vals = self.my_field_generator.evaluate_field_at_location(self.loc, xi)
        # last two arguments are relative and absolute tolerance, respectively
        np.testing.assert_allclose(
            my_vals,
            np.array(
                [
                    -0.39252351,
                    -0.59235608,
                    0.0266424,
                    0.23892774,
                    0.61143994,
                    0.92525213,
                    -0.80441616,
                ]
            ),
            1e-07,
            1e-07,
        )

    @pytest.mark.unit_tests
    def test_correlation(self):
        my_vals = np.zeros((self.loc.shape[0], 200))
        np.random.seed(self.seed)
        for i in range(200):
            xi = np.random.randn(self.my_stoch_dim, 1)
            my_vals[:, i] = self.my_field_generator.evaluate_field_at_location(self.loc, xi)

        # compute empirical correlation coefficient
        act_corr_at_dist_10_1 = np.corrcoef(my_vals[0, :], my_vals[2, :])
        act_corr_at_dist_10_2 = np.corrcoef(my_vals[0, :], my_vals[1, :])
        act_corr_at_dist_25_1 = np.corrcoef(my_vals[0, :], my_vals[4, :])
        act_corr_at_dist_25_2 = np.corrcoef(my_vals[0, :], my_vals[3, :])
        act_corr_at_dist_100_1 = np.corrcoef(my_vals[0, :], my_vals[6, :])
        act_corr_at_dist_100_2 = np.corrcoef(my_vals[0, :], my_vals[5, :])

        # expected correlation
        # exp_corr_at_dist_10     = exp(-(loc(2)-loc(1))^2/corr_length^2);
        # exp_corr_at_dist_25     = exp(-(loc(3)-loc(1))^2/corr_length^2);
        # exp_corr_at_dist_100    = exp(-(loc(4)-loc(1))^2/corr_length^2);

        # print(act_corr_at_dist_10_1[0,1])
        # print(act_corr_at_dist_10_2[0,1])
        # print(act_corr_at_dist_25_1[0,1])
        # print(act_corr_at_dist_25_2[0,1])
        # print(act_corr_at_dist_100_1[0,1])
        # print(act_corr_at_dist_100_2[0,1])

        ref_corr_at_dist_10_1 = 0.860367233062
        ref_corr_at_dist_10_2 = 0.866693990382
        ref_corr_at_dist_25_1 = 0.354716997757
        ref_corr_at_dist_25_2 = 0.340274343916
        ref_corr_at_dist_100_1 = 0.0327703467204
        ref_corr_at_dist_100_2 = -0.00267917573333

        self.assertAlmostEqual(
            act_corr_at_dist_10_1[0, 1],
            ref_corr_at_dist_10_1,
            7,
            'Correlation for distance 10 is not correct.',
        )
        self.assertAlmostEqual(
            act_corr_at_dist_10_2[0, 1],
            ref_corr_at_dist_10_2,
            7,
            'Correlation for distance 10 is not correct.',
        )
        self.assertAlmostEqual(
            act_corr_at_dist_25_1[0, 1],
            ref_corr_at_dist_25_1,
            7,
            'Correlation for distance 25 is not correct.',
        )
        self.assertAlmostEqual(
            act_corr_at_dist_25_2[0, 1],
            ref_corr_at_dist_25_2,
            7,
            'Correlation for distance 25 is not correct.',
        )
        self.assertAlmostEqual(
            act_corr_at_dist_100_1[0, 1],
            ref_corr_at_dist_100_1,
            7,
            'Correlation for distance 100 is not correct.',
        )
        self.assertAlmostEqual(
            act_corr_at_dist_100_2[0, 1],
            ref_corr_at_dist_100_2,
            7,
            'Correlation for distance 100 is not correct.',
        )

    @pytest.mark.unit_tests
    def test_marginal_distribution(self):
        my_vals = np.zeros((1, 200))
        np.random.seed(self.seed)
        for i in range(200):
            xi = np.random.randn(self.my_stoch_dim, 1)
            my_vals[0, i] = self.my_field_generator.evaluate_field_at_location(
                np.array([[25, 25]]), xi
            )

        # try to check whether marginal distribution is normally distributed
        # using kstest
        test_statistic = (stats.kstest(my_vals[0, :], 'norm'))[0]
        self.assertAlmostEqual(test_statistic, 0.041853722188287312)


if __name__ == '__main__':
    unittest.main()
