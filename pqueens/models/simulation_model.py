import numpy as np
from . model import Model
from pqueens.interfaces.interface import Interface

class SimulationModel(Model):
    """ Simulation model class """

    def __init__(self, model_name, interface, model_parameters):
        """ Initialize simulation model

        Args:
            model_name (string):        Name of model
            interface (interface):      Interface to simulator
            model_parameters (dict):    Dictionary with description of
                                        model parameters

        """
        super(SimulationModel, self).__init__(model_name, interface,
                                              model_parameters)

    @classmethod
    def from_config_create_model(cls, model_name, config):
        """  Create simulation model from problem description

        Args:
            model_name (string): Name of model
            config (dict):       Dictionary containing problem description

        Returns:
            simulation_model:   Instance of SimulationModel

        """
        # get options
        model_options = config[model_name]
        interface_name = model_options["interface"]
        parameters = model_options["parameters"]
        model_parameters = config[parameters]

        # create interface
        interface = Interface.from_config_create_interface(interface_name, config)
        return cls(model_name, interface, model_parameters)

    def evaluate(self):
        """ Evaluate model with current set of variables """
        self.response = self.interface.map(self.variables)
        return np.reshape(np.array(self.response), (-1, 1))
