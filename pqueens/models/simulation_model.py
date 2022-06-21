"""Simulation model class."""

from pqueens.interfaces import from_config_create_interface

from .model import Model


class SimulationModel(Model):
    """Simulation model class.

    Attributes:
        interface (interface):          Interface to simulations/functions
    """

    def __init__(self, model_name, interface, model_parameters):
        """Initialize simulation model.

        Args:
            model_name (string):        Name of model
            interface (interface):      Interface to simulator
            model_parameters (dict):    Dictionary with description of
                                        model parameters
        """
        super().__init__(model_name, model_parameters)
        self.interface = interface

    @classmethod
    def from_config_create_model(cls, model_name, config):
        """Create simulation model from problem description.

        Args:
            model_name (string): Name of model
            config (dict):       Dictionary containing problem description

        Returns:
            simulation_model:   Instance of SimulationModel
        """
        # get options
        model_options = config[model_name]
        interface_name = model_options["interface"]
        parameters = model_options.get("parameters")

        # get model parameters, if such parameters are provided in input
        # file (default)
        if parameters is not None:
            model_parameters = config[parameters]
        # if there are not any model parameters provided in the input file
        # (as currently for single simulation runs), generate pseudo model
        # parameters to be used below
        else:
            model_parameters = {}
            model_parameters["random_variables"] = {}
            model_parameters["random_variables"]["pseudo_var"] = {}
            model_parameters["random_variables"]["pseudo_var"]["size"] = 1
            model_parameters["random_variables"]["pseudo_var"]["type"] = 'none'

        # create interface
        interface = from_config_create_interface(interface_name, config)
        return cls(model_name, interface, model_parameters)

    def evaluate(self, gradient_bool=False):
        """Evaluate model with current set of variables.

        Args:
            gradient_bool (bool): Boolean to determine whether gradient at current variable
                                  should be evaluated as well (if True).

        Returns:
            self.response (np.array, tuple): Response of the underlying model at current variables
        """
        self.response = self.interface.evaluate(self.variables, gradient_bool=gradient_bool)
        return self.response
