import unittest

import numpy as np

from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.iterators.sobol_index_iterator import SobolIndexIterator
from pqueens.models.simulation_model import SimulationModel
from pqueens.variables.variables import Variables


class TestSobolIndices(unittest.TestCase):
    def setUp(self):

        uncertain_parameter = {
            "type": "FLOAT",
            "size": 1,
            "distribution": "uniform",
            "distribution_parameter": [-3.14159265359, 3.14159265359],
        }

        random_variables = {
            'x1': uncertain_parameter,
            'x2': uncertain_parameter,
            'x3': uncertain_parameter,
        }
        uncertain_parameters = {"random_variables": random_variables}

        some_settings = {"experiment_name": "test"}

        self.variables = Variables.from_uncertain_parameters_create(uncertain_parameters)
        self.interface = DirectPythonInterface('test_interface', 'ishigami.py', self.variables)

        # create mock model
        self.model = SimulationModel("my_model", self.interface, uncertain_parameters)

        self.my_iterator = SobolIndexIterator(
            self.model,
            seed=42,
            num_samples=3,
            calc_second_order=True,
            num_bootstrap_samples=1000,
            confidence_level=0.95,
            result_description={},
            global_settings=some_settings,
        )

    def test_correct_sampling(self):
        """Test if scaling works correctly."""
        self.my_iterator.pre_run()

        ref_vals = np.array(
            [
                [-1.7610099445, -2.5341362616, 0.1165825399],
                [1.1106020904, -2.5341362616, 0.1165825399],
                [-1.7610099445, -1.3805827091, 0.1165825399],
                [-1.7610099445, -2.5341362616, 2.5586799542],
                [-1.7610099445, -1.3805827091, 2.5586799542],
                [1.1106020904, -2.5341362616, 2.5586799542],
                [1.1106020904, -1.3805827091, 0.1165825399],
                [1.1106020904, -1.3805827091, 2.5586799542],
                [1.3805827091, 0.607456392, -3.0250101137],
                [-2.0309905632, 0.607456392, -3.0250101137],
                [1.3805827091, 1.7610099445, -3.0250101137],
                [1.3805827091, 0.607456392, -0.5829126994],
                [1.3805827091, 1.7610099445, -0.5829126994],
                [-2.0309905632, 0.607456392, -0.5829126994],
                [-2.0309905632, 1.7610099445, -3.0250101137],
                [-2.0309905632, 1.7610099445, -0.5829126994],
                [2.9513790359, -0.9633399348, 1.6873788667],
                [2.6813984172, -0.9633399348, 1.6873788667],
                [2.9513790359, 0.1902136177, 1.6873788667],
                [2.9513790359, -0.9633399348, -2.1537090262],
                [2.9513790359, 0.1902136177, -2.1537090262],
                [2.6813984172, -0.9633399348, -2.1537090262],
                [2.6813984172, 0.1902136177, 1.6873788667],
                [2.6813984172, 0.1902136177, -2.1537090262],
            ]
        )

        np.testing.assert_allclose(self.my_iterator.samples, ref_vals, 1e-07, 1e-07)

    def test_correct_sensitivity_indices(self):
        """Test if we get error when the number os distributions doas not match
        the number of parameters."""
        self.my_iterator.pre_run()
        self.my_iterator.core_run()
        si = self.my_iterator.sensitivity_indices

        ref_s1 = np.array([-0.3431210952, 1.0121533295, -0.8953575936])
        ref_s1_conf = np.array([1.90220749, 1.20190584, 0.63203508])

        ref_st = np.array([3.0745206587, 0.5874782724, 0.8353123287])
        ref_st_conf = np.array([8.65103320, 1.20338168, 1.761519808])

        ref_s2 = np.array(
            [
                [np.nan, 0.5204364349, 1.6628473529],
                [np.nan, np.nan, -0.216330859],
                [np.nan, np.nan, np.nan],
            ]
        )
        ref_s2_conf = np.array(
            [
                [np.nan, 3.42874166, 1.32548536],
                [np.nan, np.nan, 0.75616180],
                [np.nan, np.nan, np.nan],
            ]
        )

        np.testing.assert_allclose(si['S1'], ref_s1, 1e-07, 1e-07)
        np.testing.assert_allclose(si['S1_conf'], ref_s1_conf, 1e-07, 1e-07)

        np.testing.assert_allclose(si['ST'], ref_st, 1e-07, 1e-07)
        np.testing.assert_allclose(si['ST_conf'], ref_st_conf, 1e-07, 1e-07)

        np.testing.assert_allclose(si['S2'], ref_s2, 1e-07, 1e-07)
        np.testing.assert_allclose(si['S2_conf'], ref_s2_conf, 1e-07, 1e-07)
