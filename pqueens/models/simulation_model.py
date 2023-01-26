"""Simulation model class."""

from pqueens.interfaces import from_config_create_interface

from .model import Model


class SimulationModel(Model):
    """Simulation model class.

    Attributes:
        interface (interface): Interface to simulations/functions.
    """

    def __init__(self, model_name, interface):
        """Initialize simulation model.

        Args:
            model_name (string):        Name of model
            interface (interface):      Interface to simulator
        """
        super().__init__(model_name)
        self.interface = interface

    @classmethod
    def from_config_create_model(cls, model_name, config):
        """Create simulation model from problem description.

        Args:
            model_name (string): Name of model
            config (dict):       Dictionary containing problem description

        Returns:
            simulation_model: Instance of SimulationModel
        """
        # get options
        model_options = config[model_name]
        interface_name = model_options["interface_name"]

        # create interface
        interface = from_config_create_interface(interface_name, config)

        return cls(model_name, interface)

    def evaluate(self, samples):
        """Evaluate model with current set of samples.

        Args:
            samples (np.ndarray): Evaluated samples

        Returns:
            self.response (np.array): Response of the underlying model at current variables
        """
        self.response = self.interface.evaluate(samples)['mean']
        return self.response

    def _grad(self, samples, upstream):
        """Evaluate gradient of model with current set of samples.

        Args:
            samples (np.array): Evaluated samples
            upstream (np.array): Upstream gradient
        """
        raise ValueError('You have to define a Gradient model for Simulation model!')
