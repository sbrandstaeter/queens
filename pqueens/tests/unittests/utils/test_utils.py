"""Created on January 18th  2018.

@author: jbi
"""
import unittest

import mock
import numpy as np
import pytest
from sklearn.neighbors import KernelDensity

from pqueens.utils.pdf_estimation import estimate_bandwidth_for_kde, estimate_pdf

# class TestInjector(unittest.TestCase):
#     def setUp(self):
#     def test_something(self):


class TestPDFEstimation(unittest.TestCase):
    def setUp(self):
        # fix seed
        np.random.seed(42)
        # create normal random samples
        self.samples = np.random.randn(100)
        self.bandwidth = 0.4

    @pytest.mark.unit_tests
    def test_compute_kernel_bandwidth(self):
        min_samples = -3
        max_samples = 3
        bandwidth = estimate_bandwidth_for_kde(self.samples, min_samples, max_samples)
        np.testing.assert_almost_equal(bandwidth, 0.987693684111131, 7)

    @pytest.mark.unit_tests
    def test_density_estimation(self):
        supp_points = np.linspace(-1, 1, 10)
        # test with support points
        pdf_estimate, _ = estimate_pdf(self.samples, self.bandwidth, support_points=supp_points)
        pdf_desired = np.array(
            [
                0.2374548,
                0.3022495,
                0.3665407,
                0.4094495,
                0.4211095,
                0.4052543,
                0.3688492,
                0.3189032,
                0.2642803,
                0.2129013,
            ]
        )

        np.testing.assert_almost_equal(pdf_estimate, pdf_desired, 7)

        # test without support points
        pdf_estimate, supp_points = estimate_pdf(self.samples, self.bandwidth, support_points=None)

        np.testing.assert_almost_equal(pdf_estimate[50], 0.40575231779015242, 7)
        np.testing.assert_almost_equal(supp_points[50], -0.36114748358535964, 7)
