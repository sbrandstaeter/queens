'''
Created on January 18th  2018
@author: jbi

'''

import unittest
import mock
from pqueens.models.multifidelity_model import MultifidelityModel

class TestMultiFidelityModel(unittest.TestCase):
    def setUp(self):
        self.dummy_config = {"model" : {"type" : "multi_fidelity_model",
                                        "model_hierarchy" : ["lofi_borehole", "hifi_borehole"],
                                        "eval_cost_per_level" :[1, 1],
                                        "parameters" : "parameters"},
                             "parameters" : {"random_variables" :{"youngs" : {"type" : "FLOAT",
                                                               "size" : 1}}
                                            },
                             "hifi_borehole" : {"type" : "simulation_model",
                                                "interface" : "interface_hifi"
                                               },
                             "lofi_borehole" : {"type" : "simulation_model",
                                                "interface" : "interface_lofi"
                                               },
                             "interface_lofi" : {"type" : "direct_python_interface",
                                                 "main_file" : "/Users/jonas/work/adco/queens_code/pqueens/pqueens/example_simulator_functions/borehole_hifi.py"
                                                },
                             "interface_hifi" : {"type" : "direct_python_interface",
                                                 "main_file" : "/Users/jonas/work/adco/queens_code/pqueens/pqueens/example_simulator_functions/borehole_hifi.py"
                                                },
                            }


    @mock.patch('pqueens.interfaces.interface.Interface.from_config_create_interface')
    @mock.patch('pqueens.models.multifidelity_model.SimulationModel')
    def test_from_config_function(self, mock_submodel, mock_interface):
        MultifidelityModel.from_config_create_model("model", self.dummy_config)

        model_calls = [mock.call("lofi_borehole", mock_interface.return_value, self.dummy_config["parameters"]),
                       mock.call("hifi_borehole", mock_interface.return_value, self.dummy_config["parameters"])]

        mock_submodel.assert_has_calls(model_calls, any_order=False)

        interface_calls = [mock.call("interface_lofi", self.dummy_config),
                           mock.call("interface_hifi", self.dummy_config)]

        mock_interface.assert_has_calls(interface_calls, any_order=False)
