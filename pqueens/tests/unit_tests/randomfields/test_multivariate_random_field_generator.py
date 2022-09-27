"""Created on April 28th 2017.

@author: jbi
"""

import unittest

import numpy as np
import pytest
from scipy import stats
from scipy.stats import norm

from pqueens.randomfields.multivariate_random_field_generator import (
    MultiVariateRandomFieldGenerator,
)


class TestMultivariateRandomFieldGenerator(unittest.TestCase):
    def setUp(self):
        # setup some necessary variables to setup random field generators
        self.dimension = 1
        self.corrstruct = 'exp'
        self.corr_length = 25
        self.energy_frac = 0.99
        self.field_bbox = np.array([-100, 100])
        self.num_terms_per_dim = 100
        self.total_terms = 100
        self.loc = [[0], [10], [25], [100]]
        self.seed = 42
        self.num_fields = 3
        self.marginal_pdfs = []
        self.marginal_pdfs_to_many = []

        # setup marginal distributions
        for i in range(self.num_fields):
            self.marginal_pdfs.append(norm(0, 1))

        # setup cross correlation matrix
        self.crosscorr = 0.8 * np.ones((self.num_fields, self.num_fields))
        for i in range(self.num_fields):
            self.crosscorr[i, i] = 1

        # setup cross correlation matrix with wrong dimensions
        self.crosscorr_to_big = 0.8 * np.ones((self.num_fields + 1, self.num_fields + 1))
        for i in range(self.num_fields + 1):
            self.crosscorr_to_big[i, i] = 1

        # setup marginal distributions
        for i in range(self.num_fields + 1):
            self.marginal_pdfs_to_many.append(norm(0, 1))

    def test_constructor(self):
        my_field_generator = MultiVariateRandomFieldGenerator(
            marginal_distributions=self.marginal_pdfs,
            num_fields=self.num_fields,
            crosscorr=self.crosscorr,
            spatial_dimension=self.dimension,
            corr_struct=self.corrstruct,
            corr_length=self.corr_length,
            energy_frac=self.energy_frac,
            field_bbox=self.field_bbox,
            num_terms_per_dim=self.num_terms_per_dim,
            total_terms=self.total_terms,
        )
        my_stoch_dim = my_field_generator.get_stoch_dim()
        self.assertAlmostEqual(my_stoch_dim, 300)
        with self.assertRaises(RuntimeError):
            my_field_generator = MultiVariateRandomFieldGenerator(
                marginal_distributions=self.marginal_pdfs,
                num_fields=self.num_fields,
                crosscorr=self.crosscorr_to_big,
                spatial_dimension=self.dimension,
                corr_struct=self.corrstruct,
                corr_length=self.corr_length,
                energy_frac=self.energy_frac,
                field_bbox=self.field_bbox,
                num_terms_per_dim=self.num_terms_per_dim,
                total_terms=self.total_terms,
            )
        with self.assertRaises(RuntimeError):
            my_field_generator = MultiVariateRandomFieldGenerator(
                marginal_distributions=self.marginal_pdfs_to_many,
                num_fields=self.num_fields,
                crosscorr=self.crosscorr,
                spatial_dimension=self.dimension,
                corr_struct=self.corrstruct,
                corr_length=self.corr_length,
                energy_frac=self.energy_frac,
                field_bbox=self.field_bbox,
                num_terms_per_dim=self.num_terms_per_dim,
                total_terms=self.total_terms,
            )

    def test_generator(self):
        my_field_generator = MultiVariateRandomFieldGenerator(
            marginal_distributions=self.marginal_pdfs,
            num_fields=self.num_fields,
            crosscorr=self.crosscorr,
            spatial_dimension=self.dimension,
            corr_struct=self.corrstruct,
            corr_length=self.corr_length,
            energy_frac=self.energy_frac,
            field_bbox=self.field_bbox,
            num_terms_per_dim=self.num_terms_per_dim,
            total_terms=self.total_terms,
        )
        my_stoch_dim = my_field_generator.get_stoch_dim()
        np.random.seed(self.seed)
        xi = np.random.randn(my_stoch_dim, 1)
        my_vals = my_field_generator.evaluate_field_at_location(np.array([[10]]), np.array([xi]))
        ref_vals = np.array([[-1.204749293959387, -0.802391842234234, -0.801376272561015]])
        # chose lose tolerance as this differs quite a bit from machine to machine
        np.testing.assert_allclose(my_vals, ref_vals, 1e-3, 0)

    def test_cross_correlation(self):

        my_field_generator = MultiVariateRandomFieldGenerator(
            marginal_distributions=self.marginal_pdfs,
            num_fields=self.num_fields,
            crosscorr=self.crosscorr,
            spatial_dimension=self.dimension,
            corr_struct=self.corrstruct,
            corr_length=self.corr_length,
            energy_frac=self.energy_frac,
            field_bbox=self.field_bbox,
            num_terms_per_dim=self.num_terms_per_dim,
            total_terms=self.total_terms,
        )
        my_stoch_dim = my_field_generator.get_stoch_dim()
        np.random.seed(self.seed + 46453)
        my_vals = np.zeros((100, self.num_fields))
        for i in range(100):
            xi = np.random.randn(my_stoch_dim, 1)
            my_vals[i, :] = my_field_generator.evaluate_field_at_location(
                np.array([[10]]), np.array(xi)
            )

        my_cross_correlation = np.corrcoef(my_vals[:, 1], my_vals[:, 2])
        ref_cross_correlation = 0.80579072011487696
        # chose lose tolerance as this differs quite a bit from machine to machine
        np.testing.assert_allclose(my_cross_correlation[0, 1], ref_cross_correlation, 1e-1)
        # self.assertAlmostEqual(my_cross_correlation[0,1], ref_cross_correlation,
        #                           7, 'Cross correlation is not correct.')


if __name__ == '__main__':
    unittest.main()
