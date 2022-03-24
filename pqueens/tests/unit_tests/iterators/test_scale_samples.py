"""Created on November 20th  2017.

@author: jbi
"""
import unittest

import numpy as np
import pytest

from pqueens.utils.scale_samples import scale_samples


class TestScaleSamples(unittest.TestCase):
    def setUp(self):
        np.random.seed(5)
        self.num_samples = 5
        self.num_params = 3
        self.samples = np.zeros((self.num_samples, self.num_params))

        for i in range(self.num_params):
            self.samples[:, i] = np.random.uniform(0, 1, self.num_samples)

        first_distribution = {}
        first_distribution['distribution'] = "uniform"
        first_distribution['distribution_parameter'] = [-3.14159265359, 3.14159265359]

        second_distribution = {}
        second_distribution['distribution'] = "normal"
        second_distribution['distribution_parameter'] = [0, 4]

        third_distribution = {}
        third_distribution['distribution'] = "lognormal"
        third_distribution['distribution_parameter'] = [1, 2]

        self.distribution_info_to_small = [first_distribution, second_distribution]

        self.distribution_info = [first_distribution, second_distribution, third_distribution]

        self.distribution_info_to_big = [
            first_distribution,
            second_distribution,
            third_distribution,
            third_distribution,
        ]

        # setup some other distribution dicts with unadmissible distribution
        # parameters
        wrong_uniform_distribution = {}
        wrong_uniform_distribution['distribution'] = "uniform"
        wrong_uniform_distribution['distribution_parameter'] = [0, -2]

        wrong_normal_distribution = {}
        wrong_normal_distribution['distribution'] = "normal"
        wrong_normal_distribution['distribution_parameter'] = [0, -4]

        wrong_lognormal_distribution = {}
        wrong_lognormal_distribution['distribution'] = "lognormal"
        wrong_lognormal_distribution['distribution_parameter'] = [0, -2]

        self.wrong_uniform_distribution_list = [wrong_uniform_distribution] * 3
        self.wrong_normal_distribution_list = [wrong_normal_distribution] * 3
        self.wrong_lognormal_distribution_list = [wrong_lognormal_distribution] * 3

    @pytest.mark.unit_tests
    def test_correct_sample_scaling(self):
        """Test if scaling works correctly."""

        ref_vals = np.array(
            [
                [-1.74676842, 0.56773409, 0.16526491],
                [2.32937978, 1.450873, 9.74825854],
                [-1.84273789, 0.09236692, 2.02323682],
                [2.63020991, -1.06725001, 0.36682998],
                [-0.07281465, -1.77264987, 28.48404597],
            ]
        )

        scaled_samples = scale_samples(self.samples, self.distribution_info)
        # print("shape scale_samples: {}".format(scaled_samples))
        np.testing.assert_allclose(scaled_samples, ref_vals, 1e-07, 1e-07)

    @pytest.mark.unit_tests
    def test_wrong_distribution_parameters(self):
        """Test if scale samples trows error if we pass unadmissible
        distribution parameters."""

        with self.assertRaises(ValueError):
            scale_samples(self.samples, self.wrong_uniform_distribution_list)
        with self.assertRaises(ValueError):
            scale_samples(self.samples, self.wrong_normal_distribution_list)
        with self.assertRaises(ValueError):
            scale_samples(self.samples, self.wrong_lognormal_distribution_list)

    @pytest.mark.unit_tests
    def test_non_matching_inputs(self):
        """Test if we get error when the number os distributions doas not match
        the number of parameters."""

        with self.assertRaises(ValueError):
            scale_samples(self.samples, self.distribution_info_to_small)
        with self.assertRaises(ValueError):
            scale_samples(self.samples, self.distribution_info_to_big)
