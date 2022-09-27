"""Created on November 23th  2017.

@author: jbi
"""
import unittest

import numpy as np
import pytest

import pqueens.parameters.parameters as parameters_module
from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.iterators.lhs_iterator import LHSIterator
from pqueens.models.simulation_model import SimulationModel
from pqueens.tests.integration_tests.example_simulator_functions import (
    example_simulator_function_by_name,
)


class TestLHSIterator(unittest.TestCase):
    """Test LHS Iterator."""

    def setUp(self):
        random_variables = {}
        uncertain_parameters = {}
        uncertain_parameter1 = {}
        uncertain_parameter1["type"] = "FLOAT"
        uncertain_parameter1["dimension"] = 1
        uncertain_parameter1["distribution"] = "uniform"
        uncertain_parameter1["lower_bound"] = -3.14159265359
        uncertain_parameter1["upper_bound"] = 3.14159265359

        uncertain_parameter2 = {}
        uncertain_parameter2["type"] = "FLOAT"
        uncertain_parameter2["dimension"] = 1
        uncertain_parameter2["distribution"] = "normal"
        uncertain_parameter2["mean"] = 0
        uncertain_parameter2["covariance"] = 4

        uncertain_parameter3 = {}
        uncertain_parameter3["type"] = "FLOAT"
        uncertain_parameter3["dimension"] = 1
        uncertain_parameter3["distribution"] = "lognormal"
        uncertain_parameter3["normal_mean"] = 0.3
        uncertain_parameter3["normal_covariance"] = 1

        random_variables['x1'] = uncertain_parameter1
        random_variables['x2'] = uncertain_parameter2
        random_variables['x3'] = uncertain_parameter3
        uncertain_parameters["random_variables"] = random_variables

        parameters_module.from_config_create_parameters({"parameters": uncertain_parameters})

        some_settings = {}
        some_settings["experiment_name"] = "test"

        function = example_simulator_function_by_name("ishigami90")
        # create interface
        self.interface = DirectPythonInterface('test_interface', function, pool=None)

        # create mock model
        self.model = SimulationModel("my_model", self.interface)

        # create LHS iterator
        self.my_iterator = LHSIterator(
            self.model,
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
        # print("Samples first row {}".format(self.my_iterator.samples[0,:]))
        # print("Sample mean {}".format(my_means))
        # print("Sample std {}".format(my_std))
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
        # print("Results first 10 {}".format(self.my_iterator.outputs[0:10]))

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
