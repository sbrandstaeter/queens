"""Unit tests for LHS iterator."""
import unittest

import numpy as np

from pqueens.distributions.lognormal import LogNormalDistribution
from pqueens.distributions.normal import NormalDistribution
from pqueens.distributions.uniform import UniformDistribution
from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.iterators.lhs_iterator import LHSIterator
from pqueens.models.simulation_model import SimulationModel
from pqueens.parameters.parameters import Parameters


class TestLHSIterator(unittest.TestCase):
    """Test LHS Iterator."""

    def setUp(self):
        """TODO_doc."""
        x1 = UniformDistribution(lower_bound=-3.14159265359, upper_bound=3.14159265359)
        x2 = NormalDistribution(mean=0, covariance=4)
        x3 = LogNormalDistribution(normal_mean=0.3, normal_covariance=1)
        parameters = Parameters(x1=x1, x2=x2, x3=x3)

        some_settings = {"experiment_name": "test"}

        # create interface
        self.interface = DirectPythonInterface(
            parameters=parameters, function="ishigami90", num_workers=1
        )

        # create mock model
        self.model = SimulationModel(self.interface)

        # create LHS iterator
        self.my_iterator = LHSIterator(
            self.model,
            parameters=parameters,
            seed=42,
            num_samples=100,
            num_iterations=1,
            result_description=None,
            global_settings=some_settings,
            criterion='maximin',
        )

    def test_correct_sampling(self):
        """Test if we get correct samples."""
        # np.set_printoptions(precision=10)
        self.my_iterator.pre_run()

        # check if mean and std match
        means_ref = np.array([-1.4546056001e-03, 5.4735307403e-03, 2.1664850171e00])

        np.testing.assert_allclose(
            np.mean(self.my_iterator.samples, axis=0), means_ref, 1e-09, 1e-09
        )

        std_ref = np.array([1.8157451781, 1.9914892803, 2.4282341125])
        np.testing.assert_allclose(np.std(self.my_iterator.samples, axis=0), std_ref, 1e-09, 1e-09)

        # check if samples are identical too
        ref_sample_first_row = np.array([-2.7374616292, -0.6146554017, 1.3925529817])

        np.testing.assert_allclose(
            self.my_iterator.samples[0, :], ref_sample_first_row, 1e-07, 1e-07
        )

    def test_correct_results(self):
        """Test if we get correct results."""
        self.my_iterator.pre_run()
        self.my_iterator.core_run()

        # np.set_printoptions(precision=10)

        # check if samples are identical too
        ref_results = np.array(
            [
                [1.7868040337],
                [-13.8624183835],
                [6.3423271929],
                [6.1674472752],
                [5.3528917433],
                [-0.7472766806],
                [5.0007066283],
                [6.4763926539],
                [-6.4173504897],
                [3.1739282221],
            ]
        )

        np.testing.assert_allclose(self.my_iterator.output["mean"][0:10], ref_results, 1e-09, 1e-09)
