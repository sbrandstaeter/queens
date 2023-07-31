"""Unit test for Sobol index iterator."""
import unittest
from copy import deepcopy

import numpy as np

from pqueens.distributions.uniform import UniformDistribution
from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.iterators.sobol_index_iterator import SobolIndexIterator
from pqueens.models.simulation_model import SimulationModel
from pqueens.parameters.parameters import Parameters


class TestSobolIndices(unittest.TestCase):
    """Unit test for Sobol index iterator."""

    def setUp(self):
        """Set up the iterator."""
        rv = UniformDistribution(lower_bound=-3.14159265359, upper_bound=3.14159265359)
        parameters = Parameters(x1=rv, x2=deepcopy(rv), x3=deepcopy(rv))
        some_settings = {"experiment_name": "test"}

        self.interface = DirectPythonInterface(
            parameters=parameters, function="ishigami90", num_workers=1
        )

        # create mock model
        self.model = SimulationModel(self.interface)

        self.my_iterator = SobolIndexIterator(
            self.model,
            parameters=parameters,
            seed=42,
            num_samples=2,
            calc_second_order=True,
            num_bootstrap_samples=1000,
            confidence_level=0.95,
            result_description={},
            global_settings=some_settings,
        )

    def test_correct_sampling(self):
        """Test correct sampling."""
        self.my_iterator.pre_run()

        ref_vals = np.array(
            [
                [-3.1323887689, -0.7761942787, -0.3282718886],
                [-0.0828349625, -0.7761942787, -0.3282718886],
                [-3.1323887689, 0.3589515044, -0.3282718886],
                [-3.1323887689, -0.7761942787, 2.1629129109],
                [-3.1323887689, 0.3589515044, 2.1629129109],
                [-0.0828349625, -0.7761942787, 2.1629129109],
                [-0.0828349625, 0.3589515044, -0.3282718886],
                [-0.0828349625, 0.3589515044, 2.1629129109],
                [0.0092038847, 2.3653983749, 2.8133207650],
                [3.0587576910, 2.3653983749, 2.8133207650],
                [0.0092038847, -2.7826411492, 2.8133207650],
                [0.0092038847, 2.3653983749, -0.9786797427],
                [0.0092038847, -2.7826411492, -0.9786797427],
                [3.0587576910, 2.3653983749, -0.9786797427],
                [3.0587576910, -2.7826411492, 2.8133207650],
                [3.0587576910, -2.7826411492, -0.9786797427],
            ]
        )

        np.testing.assert_allclose(self.my_iterator.samples, ref_vals, 1e-07, 1e-07)

    def test_correct_sensitivity_indices(self):
        """Test Sobol indices results."""
        self.my_iterator.pre_run()
        self.my_iterator.core_run()
        si = self.my_iterator.sensitivity_indices

        ref_s1 = np.array([-0.1500205418, 2.0052373431, 0.0282147215])
        ref_s1_conf = np.array([0.3189669552, 0.0599864170, 0.0186884010])

        ref_st = np.array([0.0399037376, 1.8151878529, 0.0004985237])
        ref_st_conf = np.array([0.0589769471, 0.2607858609, 0.0006083827])

        ref_s2 = np.array(
            [
                [np.nan, 0.0915118400, 0.2003003441],
                [np.nan, np.nan, -0.2712666971],
                [np.nan, np.nan, np.nan],
            ]
        )
        ref_s2_conf = np.array(
            [
                [np.nan, 0.4690527908, 0.1802601772],
                [np.nan, np.nan, 0.3460941042],
                [np.nan, np.nan, np.nan],
            ]
        )

        np.testing.assert_allclose(si['S1'], ref_s1, 1e-07, 1e-07)
        np.testing.assert_allclose(si['S1_conf'], ref_s1_conf, 1e-07, 1e-07)

        np.testing.assert_allclose(si['ST'], ref_st, 1e-07, 1e-07)
        np.testing.assert_allclose(si['ST_conf'], ref_st_conf, 1e-07, 1e-07)

        np.testing.assert_allclose(si['S2'], ref_s2, 1e-07, 1e-07)
        np.testing.assert_allclose(si['S2_conf'], ref_s2_conf, 1e-07, 1e-07)
