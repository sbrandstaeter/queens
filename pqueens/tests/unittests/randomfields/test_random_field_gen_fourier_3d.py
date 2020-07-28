'''
Created on April 21th 2017
@author: jbi

'''

import unittest
import numpy as np

from scipy import stats
from scipy.stats import norm
from pqueens.randomfields.random_field_gen_fourier_3d import RandomFieldGenFourier3D
from pqueens.randomfields.univariate_field_generator_factory import (
    UniVarRandomFieldGeneratorFactory,
)


class TestRandomFieldGeneratorFourier3D(unittest.TestCase):
    def setUp(self):
        # setup some necessary variables to setup random field generators
        self.dimension = 3
        self.corrstruct = 'squared_exp'
        self.corr_length = 25
        self.energy_frac = 0.95
        self.field_bbox = np.array([-100, 100, -100, 100, -100, 100])
        self.marginal_pdf = norm(0, 1)
        self.num_terms_per_dim = 20
        self.total_terms = 40
        self.loc = np.array(
            [
                [0, 0, 0],
                [0, 0, 10],
                [0, 10, 0],
                [10, 0, 0],
                [0, 0, 25],
                [0, 25, 0],
                [25, 0, 0],
                [0, 0, 100],
                [0, 100, 0],
                [100, 0, 0],
            ]
        )
        self.seed = 42

        # pylint: disable=line-too-long
        self.my_field_generator = UniVarRandomFieldGeneratorFactory.create_new_random_field_generator(
            marg_pdf=self.marginal_pdf,
            spatial_dimension=self.dimension,
            corrstruct=self.corrstruct,
            corr_length=self.corr_length,
            energy_frac=self.energy_frac,
            field_bbox=self.field_bbox,
            num_terms_per_dim=self.num_terms_per_dim,
            total_terms=self.total_terms,
        )
        # pylint: enable=line-too-long

        self.my_stoch_dim = self.my_field_generator.get_stoch_dim()

    # should trigger error because desired energy fraction not reached
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
            mystuff.gen_sample_gauss_field(np.array([[10, 10, 10]]), np.array((4, 4)))

    # should trigger error because dimension of location is of
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
            mystuff.gen_sample_gauss_field(np.array([[10, 10]]), xi)

    def test_values_at_location(self):
        np.random.seed(self.seed)
        xi = np.random.randn(self.my_stoch_dim, 1)
        my_vals = self.my_field_generator.evaluate_field_at_location(self.loc, xi)

        # np.set_printoptions(formatter={'float': '{: 0.15f}'.format})
        # print(my_vals)

        ref_vals = np.array(
            [
                -0.272606175022982,
                -0.364627863004368,
                0.126620737359761,
                0.460434864806447,
                0.293453869499679,
                -0.156924505159762,
                0.310580566207436,
                -0.431044039109896,
                0.265531834295388,
                -0.853659094570138,
            ]
        )
        # last two arguments are relative and absolute tolerance, respectively
        np.testing.assert_allclose(my_vals, ref_vals, 1e-10, 1e-10)

    def test_correlation(self):
        my_vals = np.zeros((self.loc.shape[0], 200))
        np.random.seed(self.seed)
        for i in range(200):
            xi = np.random.randn(self.my_stoch_dim, 1)
            my_vals[:, i] = self.my_field_generator.evaluate_field_at_location(self.loc, xi)

        # compute empirical correlation coefficient
        act_corr_at_dist_10_1 = np.corrcoef(my_vals[0, :], my_vals[3, :])
        act_corr_at_dist_10_2 = np.corrcoef(my_vals[0, :], my_vals[2, :])
        act_corr_at_dist_10_3 = np.corrcoef(my_vals[0, :], my_vals[1, :])

        act_corr_at_dist_25_1 = np.corrcoef(my_vals[0, :], my_vals[6, :])
        act_corr_at_dist_25_2 = np.corrcoef(my_vals[0, :], my_vals[5, :])
        act_corr_at_dist_25_3 = np.corrcoef(my_vals[0, :], my_vals[4, :])

        act_corr_at_dist_100_1 = np.corrcoef(my_vals[0, :], my_vals[7, :])
        act_corr_at_dist_100_2 = np.corrcoef(my_vals[0, :], my_vals[8, :])
        act_corr_at_dist_100_3 = np.corrcoef(my_vals[0, :], my_vals[9, :])

        # expected correlation
        # exp_corr_at_dist_10     = exp(-(loc(2)-loc(1))^2/corr_length^2);
        # exp_corr_at_dist_25     = exp(-(loc(3)-loc(1))^2/corr_length^2);
        # exp_corr_at_dist_100    = exp(-(loc(4)-loc(1))^2/corr_length^2);

        # np.set_printoptions(formatter={'float': '{: 0.15f}'.format})
        # print(act_corr_at_dist_10_1[0,1])
        # print(act_corr_at_dist_10_2[0,1])
        # print(act_corr_at_dist_10_3[0,1])
        #
        # print(act_corr_at_dist_25_1[0,1])
        # print(act_corr_at_dist_25_2[0,1])
        # print(act_corr_at_dist_25_3[0,1])
        #
        # print(act_corr_at_dist_100_1[0,1])
        # print(act_corr_at_dist_100_2[0,1])
        # print(act_corr_at_dist_100_3[0,1])

        ref_corr_at_dist_10_1 = 0.781794339099
        ref_corr_at_dist_10_2 = 0.82360677963
        ref_corr_at_dist_10_3 = 0.808825341582

        ref_corr_at_dist_25_1 = 0.271614995918
        ref_corr_at_dist_25_2 = 0.351525146945
        ref_corr_at_dist_25_3 = 0.255857455956

        ref_corr_at_dist_100_1 = -0.0559324929612
        ref_corr_at_dist_100_2 = -0.0273227649886
        ref_corr_at_dist_100_3 = -0.0514774377254

        self.assertAlmostEqual(
            act_corr_at_dist_10_1[0, 1],
            ref_corr_at_dist_10_1,
            9,
            'Correlation for distance 10 is not correct.',
        )
        self.assertAlmostEqual(
            act_corr_at_dist_10_2[0, 1],
            ref_corr_at_dist_10_2,
            9,
            'Correlation for distance 10 is not correct.',
        )
        self.assertAlmostEqual(
            act_corr_at_dist_10_3[0, 1],
            ref_corr_at_dist_10_3,
            9,
            'Correlation for distance 10 is not correct.',
        )

        self.assertAlmostEqual(
            act_corr_at_dist_25_1[0, 1],
            ref_corr_at_dist_25_1,
            9,
            'Correlation for distance 25 is not correct.',
        )
        self.assertAlmostEqual(
            act_corr_at_dist_25_2[0, 1],
            ref_corr_at_dist_25_2,
            9,
            'Correlation for distance 25 is not correct.',
        )
        self.assertAlmostEqual(
            act_corr_at_dist_25_3[0, 1],
            ref_corr_at_dist_25_3,
            9,
            'Correlation for distance 25 is not correct.',
        )
        self.assertAlmostEqual(
            act_corr_at_dist_100_1[0, 1],
            ref_corr_at_dist_100_1,
            9,
            'Correlation for distance 100 is not correct.',
        )
        self.assertAlmostEqual(
            act_corr_at_dist_100_2[0, 1],
            ref_corr_at_dist_100_2,
            9,
            'Correlation for distance 100 is not correct.',
        )
        self.assertAlmostEqual(
            act_corr_at_dist_100_3[0, 1],
            ref_corr_at_dist_100_3,
            9,
            'Correlation for distance 100 is not correct.',
        )

    def test_marginal_distribution(self):
        my_vals = np.zeros((1, 200))
        np.random.seed(self.seed)
        for i in range(200):
            xi = np.random.randn(self.my_stoch_dim, 1)
            my_vals[0, i] = self.my_field_generator.evaluate_field_at_location(
                np.array([[25, 25, 25]]), xi
            )

        # try to check whether marginal distribution is normally distributed
        # using kstest
        test_statistic = (stats.kstest(my_vals[0, :], 'norm'))[0]
        # print((stats.kstest(my_vals[0, :], 'norm')))
        self.assertAlmostEqual(test_statistic, 0.041586864909728682)


if __name__ == '__main__':
    unittest.main()
