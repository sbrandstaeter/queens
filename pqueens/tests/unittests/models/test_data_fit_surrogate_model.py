'''
Created on January 18th  2018
@author: jbi

'''

import unittest
import mock
import pytest
from pqueens.models.data_fit_surrogate_model import DataFitSurrogateModel


class TestDataFitSurrogateModel(unittest.TestCase):
    def setUp(self):
        self.dummy_config = {
            "model": {
                "type": "simulation_model",
                "interface": "dummy_interface",
                "subordinate_model": "dummy_model",
                "eval_fit": "kfold",
                "error_measures": ["sum_squared"],
                "subordinate_iterator": "dummy_iterator",
                "parameters": "dummy_parameters",
            },
            "dummy_parameters": {
                "random_variables": {
                    "youngs": {
                        "type": "FLOAT",
                        "size": 1,
                        "min": 1000000,
                        "max": 5000000,
                        "distribution": "normal",
                        "distribution_parameter": [400000, 100000000],
                    }
                }
            },
        }

    @pytest.mark.unit_tests
    @mock.patch('pqueens.iterators.iterator.Iterator.from_config_create_iterator')
    @mock.patch('pqueens.interfaces.interface.Interface.from_config_create_interface')
    @mock.patch('pqueens.models.model.Model.from_config_create_model')
    def test_from_config_function(self, mock_submodel, mock_interface, mock_iterator):
        DataFitSurrogateModel.from_config_create_model("model", self.dummy_config)
        mock_iterator.assert_called_with(
            self.dummy_config, "dummy_iterator", mock_submodel.return_value
        )
        mock_submodel.assert_called_with("dummy_model", self.dummy_config)
        mock_interface.assert_called_with("dummy_interface", self.dummy_config)
