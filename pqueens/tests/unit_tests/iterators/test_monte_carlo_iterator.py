"""Created on November 23th  2017.

@author: jbi
"""
import unittest

import numpy as np

import pqueens.parameters.parameters as parameters_module
from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.iterators.monte_carlo_iterator import MonteCarloIterator
from pqueens.models.simulation_model import SimulationModel
from pqueens.tests.integration_tests.example_simulator_functions import (
    example_simulator_function_by_name,
)


class TestMCIterator(unittest.TestCase):
    """TODO_doc."""

    def setUp(self):
        """TODO_doc."""
        random_variables = {}
        uncertain_parameter1 = {}
        uncertain_parameter1["type"] = "uniform"
        uncertain_parameter1["lower_bound"] = -3.14159265359
        uncertain_parameter1["upper_bound"] = 3.14159265359

        uncertain_parameter2 = {}
        uncertain_parameter2["type"] = "normal"
        uncertain_parameter2["mean"] = 0
        uncertain_parameter2["covariance"] = 4

        uncertain_parameter3 = {}
        uncertain_parameter3["type"] = "lognormal"
        uncertain_parameter3["normal_mean"] = 0.3
        uncertain_parameter3["normal_covariance"] = 1

        random_variables['x1'] = uncertain_parameter1
        random_variables['x2'] = uncertain_parameter2
        random_variables['x3'] = uncertain_parameter3
        some_settings = {}
        some_settings["experiment_name"] = "test"

        parameters_module.from_config_create_parameters({"parameters": random_variables})

        function = example_simulator_function_by_name("ishigami90")
        # create interface
        self.interface = DirectPythonInterface('test_interface', function, pool=None)

        # create mock model
        self.model = SimulationModel("my_model", self.interface)

        # create LHS iterator
        self.my_iterator = MonteCarloIterator(
            self.model,
            seed=42,
            num_samples=100,
            result_description=None,
            global_settings=some_settings,
        )

    def test_correct_sampling(self):
        """Test if we get correct samples."""
        self.my_iterator.pre_run()

        # check if mean and std match
        means_ref = np.array([-1.8735991508e-01, -2.1607203347e-03, 2.8955130234e00])

        np.testing.assert_allclose(
            np.mean(self.my_iterator.samples, axis=0), means_ref, 1e-09, 1e-09
        )

        std_ref = np.array([1.8598117085, 1.8167064845, 6.7786919771])
        np.testing.assert_allclose(np.std(self.my_iterator.samples, axis=0), std_ref, 1e-09, 1e-09)

        # check if samples are identical too
        ref_sample_first_row = np.array([-0.7882876819, 0.1740941365, 1.3675241182])

        np.testing.assert_allclose(
            self.my_iterator.samples[0, :], ref_sample_first_row, 1e-07, 1e-07
        )

    def test_correct_results(self):
        """Test if we get correct results."""
        self.my_iterator.pre_run()
        self.my_iterator.core_run()

        ref_results = np.array(
            [
                [-7.4713449052e-01],
                [3.6418728120e01],
                [1.3411821745e00],
                [1.0254005782e04],
                [-2.9330095397e00],
                [2.1639496168e00],
                [-1.1964201899e-01],
                [7.6345947125e00],
                [7.6591139616e00],
                [1.1519434320e01],
            ]
        )

        np.testing.assert_allclose(self.my_iterator.output["mean"][0:10], ref_results, 1e-09, 1e-09)
