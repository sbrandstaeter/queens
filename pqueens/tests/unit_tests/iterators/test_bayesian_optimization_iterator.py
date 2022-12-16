"""Created on February 6th  2018.

@author: jbi
"""
# import numpy as np
# import unittest
# import mock
# from pqueens.variables.variables import Variables
# from pqueens.iterators.bayesian_optimization_iterator import BayesOptIterator

# def side_effect(X):
#    return np.ones(len(X)).reshape(-1, 1)


# class TestBayesianOptimizationIterator(unittest.TestCase):
#    def setUp(self):
#        self.dummy_config = {
#            "global_settings": {"experiment_name": "test", "output_dir": "/dummy/dir"},
#            "method": {
#                "type": "bayesian_optimization",
#                "seed": 42,
#                "model_name": "model",
#                "num_iter": 10,
#                "use_ard": "True",
#                "num_initial_samples": 5,
#            },
#            "model": {
#                "type": "simulation_model",
#                "interface_name": "dummy_interface",
#                "parameters": "dummy_parameters",
#            },
#            "dummy_parameters": {
#                "random_variables": {
#                    "youngs": {
#                        "type": "FLOAT",
#                        "dimension": 1,
#                        "lower_bound": 1000000,
#                        "upper_bound": 5000000,
#                        "distribution": "normal",
#                        "distribution_parameter": [400000, 100000000],
#                    }
#                }
#            },
#        }
#        self.dummy_params = {
#            "random_variables": {
#                "youngs": {
#                    "type": "FLOAT",
#                    "dimension": 1,
#                    "lower_bound": 1000,
#                    "upper_bound": 5000000,
#                    "distribution": "uniform",
#                    "distribution_parameter": [1000, 10000],
#                }
#            }
#        }


# @mock.patch('pqueens.models.model.Model.from_config_create_model')
# @mock.patch(
#    'pqueens.iterators.bayesian_optimization_iterator.BayesOptIterator.random_fields',
#    return_value=None,
# )
# def test_from_config_function(self, mock_get_parameters, mock_model):
#    BayesOptIterator.from_config_create_iterator(self.dummy_config)
#    mock_model.assert_called_with("model", self.dummy_config)
#
#
# @mock.patch('pqueens.models.simulation_model.SimulationModel')
# @mock.patch(
#    'pqueens.iterators.bayesian_optimization_iterator.BayesOptIterator.prep_and_eval_model',
#    side_effect=side_effect,
# )
# @mock.patch('pqueens.iterators.bayesian_optimization_iterator.BayesianOptimizer.optimize')
# def test_core_run(self, mocks, mock2, mock3):
#    mocks.get_parameter.return_value = self.dummy_params
#    some_settings = {}
#    some_settings["experiment_name"] = "test"
#    my_iterator = BayesOptIterator(
#        mocks,
#        seed=42,
#        num_iter=10,
#        use_ard="True",
#        num_initial_samples=5,
#        result_description=None,
#        global_settings=some_settings,
#    )
#    my_iterator.core_run()
#    np.testing.assert_array_equal(
#        np.array([[3250], [5500], [1000], [7750], [10000]]), mock2.call_args[0][0]
#    )
#
# assert the BayesianOptimizer has been called the correctly
# TODO assert the BayesianOptimizer has been called the correctly
