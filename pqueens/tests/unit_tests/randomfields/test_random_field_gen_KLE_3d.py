"""Created on April 25th 2017.

@author: jbi
"""

import unittest

import numpy as np
from scipy import stats
from scipy.stats import norm

from pqueens.randomfields.univariate_field_generator_factory import (
    UniVarRandomFieldGeneratorFactory,
)


class TestRandomFieldGeneratorKLE3D(unittest.TestCase):
    """TODO_doc."""

    def setUp(self):
        """TODO_doc."""
        # setup some necessary variables to setup random field generators
        self.dimension = 3
        self.corrstruct = 'exp'
        self.corr_length = 25
        self.energy_frac = 0.95
        self.field_bbox = np.array([-100, 100, -100, 100, -100, 100])
        self.marginal_pdf = norm(0, 1)
        self.num_terms_per_dim = 200
        self.total_terms = 10000
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
    def test_not_enough_fourier_terms(self):
        """TODO_doc."""
        with self.assertRaises(RuntimeError):
            UniVarRandomFieldGeneratorFactory.create_new_random_field_generator(
                marg_pdf=self.marginal_pdf,
                spatial_dimension=self.dimension,
                corrstruct=self.corrstruct,
                corr_length=self.corr_length,
                energy_frac=0.95,
                field_bbox=self.field_bbox,
                num_terms_per_dim=100,
                total_terms=5000,
            )

    # # should trigger error because number of phase angles do not match stochastic
    # # dimension
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
            mystuff.gen_sample_gauss_field(np.array([[10, 10, 10]]), np.array((4, 4)))

    # # should trigger error because dimension of location is wrong
    def test_wrong_number_loc_dimensions(self):
        """TODO_doc."""
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
        xi = np.random.randn(self.my_stoch_dim, 1)
        with self.assertRaises(RuntimeError):
            mystuff.gen_sample_gauss_field(np.array(([4, 4, 4, 4], [3, 4, 4, 4])), xi)
        with self.assertRaises(RuntimeError):
            mystuff.gen_sample_gauss_field(np.array(([4], [4], [4])), xi)
        with self.assertRaises(RuntimeError):
            mystuff.gen_sample_gauss_field(np.array(([4, 4], [4, 5], [4, 5])), xi)

    def test_values_at_location(self):
        """TODO_doc."""
        np.random.seed(self.seed)
        xi = np.random.randn(self.my_stoch_dim, 1)

        my_vals = self.my_field_generator.evaluate_field_at_location(self.loc, xi)

        np.testing.assert_allclose(
            my_vals,
            np.array(
                [
                    [-0.267680380989928],
                    [-0.764019048145033],
                    [0.812101592243011],
                    [0.506503220813194],
                    [-0.722592329059823],
                    [-0.034977903350059],
                    [0.275229053348657],
                    [0.611625601332445],
                    [0.483895701080925],
                    [0.900676537769744],
                ]
            ),
            1e-07,
            1e-07,
        )

    def test_correlation(self):
        """TODO_doc."""
        my_vals = np.zeros((self.loc.shape[0], 100))
        np.random.seed(self.seed)
        for i in range(100):
            xi = np.random.randn(self.my_stoch_dim, 1)
            my_vals[:, i] = self.my_field_generator.evaluate_field_at_location(self.loc, xi).ravel()

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

        ref_corr_at_dist_10_1 = 0.718169328847
        ref_corr_at_dist_10_2 = 0.764240432309
        ref_corr_at_dist_10_3 = 0.720324288459

        ref_corr_at_dist_25_1 = 0.518546711936
        ref_corr_at_dist_25_2 = 0.300828372072
        ref_corr_at_dist_25_3 = 0.329607881848

        ref_corr_at_dist_100_1 = 0.006476671345
        ref_corr_at_dist_100_2 = 0.0994126929964
        ref_corr_at_dist_100_3 = 0.0425402198459

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
        """TODO_doc."""
        my_vals = np.zeros((1, 200))
        np.random.seed(self.seed)
        for i in range(200):
            xi = np.random.randn(self.my_stoch_dim, 1)
            my_vals[0, i] = self.my_field_generator.evaluate_field_at_location(
                np.array([[30, 30, 30]]), xi
            )

        # try to check whether marginal distribution is normally distributed
        # using kstest
        test_statistic = (stats.kstest(my_vals[0, :], 'norm'))[0]
        self.assertAlmostEqual(test_statistic, 0.053673663659756343)


if __name__ == '__main__':
    unittest.main()
