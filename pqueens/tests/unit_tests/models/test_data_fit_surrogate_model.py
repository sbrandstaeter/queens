"""Created on January 18th  2018.

@author: jbi
"""

import unittest

import mock
import pytest

from pqueens.models.data_fit_surrogate_model import DataFitSurrogateModel


class TestDataFitSurrogateModel(unittest.TestCase):
    """Unit tests related to datafit surrogates."""

    def setUp(self):
        """Set up test."""
        self.dummy_config = {
            "model": {
                "type": "simulation_model",
                "interface_name": "dummy_interface",
                "subordinate_model_name": "dummy_model",
                "eval_fit": "kfold",
                "error_measures": ["sum_squared"],
                "subordinate_iterator_name": "dummy_iterator",
                "parameters": "dummy_parameters",
            },
            "dummy_parameters": {
                "youngs": {
                    "type": "normal",
                    "mean": 1000000,
                    "covariance": 5000000,
                }
            },
        }

    @pytest.mark.unit_tests
    @mock.patch('pqueens.models.data_fit_surrogate_model.from_config_create_iterator')
    @mock.patch('pqueens.models.data_fit_surrogate_model.from_config_create_interface')
    @mock.patch('pqueens.models.data_fit_surrogate_model.from_config_create_model')
    def test_from_config_function(self, mock_submodel, mock_interface, mock_iterator):
        """Test the fcc function."""
        DataFitSurrogateModel.from_config_create_model("model", self.dummy_config)
        mock_iterator.assert_called_with(
            self.dummy_config, "dummy_iterator", mock_submodel.return_value
        )
        mock_submodel.assert_called_with("dummy_model", self.dummy_config)
        mock_interface.assert_called_with("dummy_interface", self.dummy_config)
