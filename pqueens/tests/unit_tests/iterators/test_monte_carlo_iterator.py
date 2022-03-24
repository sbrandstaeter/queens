"""Created on November 23th  2017.

@author: jbi
"""
import unittest

import numpy as np
import pytest

from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.iterators.monte_carlo_iterator import MonteCarloIterator
from pqueens.models.simulation_model import SimulationModel
from pqueens.variables.variables import Variables


class TestMCIterator(unittest.TestCase):
    def setUp(self):
        random_variables = {}
        uncertain_parameters = {}
        uncertain_parameter1 = {}
        uncertain_parameter1["type"] = "FLOAT"
        uncertain_parameter1["size"] = 1
        uncertain_parameter1["distribution"] = "uniform"
        uncertain_parameter1["distribution_parameter"] = [-3.14159265359, 3.14159265359]

        uncertain_parameter2 = {}
        uncertain_parameter2["type"] = "FLOAT"
        uncertain_parameter2["size"] = 1
        uncertain_parameter2["distribution"] = "normal"
        uncertain_parameter2["distribution_parameter"] = [0, 4]

        uncertain_parameter3 = {}
        uncertain_parameter3["type"] = "FLOAT"
        uncertain_parameter3["size"] = 1
        uncertain_parameter3["distribution"] = "lognormal"
        uncertain_parameter3["distribution_parameter"] = [0.3, 1]

        random_variables['x1'] = uncertain_parameter1
        random_variables['x2'] = uncertain_parameter2
        random_variables['x3'] = uncertain_parameter3
        uncertain_parameters["random_variables"] = random_variables
        some_settings = {}
        some_settings["experiment_name"] = "test"
        dummy_obj = None
        dummy_db = None

        self.variables = Variables.from_uncertain_parameters_create(uncertain_parameters)

        # create interface
        self.interface = DirectPythonInterface('test_interface', 'ishigami.py', self.variables)

        # create mock model
        self.model = SimulationModel("my_model", self.interface, uncertain_parameters)

        # create LHS iterator
        self.my_iterator = MonteCarloIterator(
            self.model,
            seed=42,
            num_samples=100,
            result_description=None,
            global_settings=some_settings,
            external_geometry_obj=dummy_obj,
            db=dummy_db,
        )

    @pytest.mark.unit_tests
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

    @pytest.mark.unit_tests
    def test_correct_results(self):
        """Test if we get correct results."""
        self.my_iterator.pre_run()
        self.my_iterator.core_run()

        # np.set_printoptions(precision=10)
        # print("Results first 10 {}".format(self.my_iterator.outputs[0:10]))

        # check if samples are identical too
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
