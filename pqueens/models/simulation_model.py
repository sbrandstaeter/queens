"""Simulation model class."""
import numpy as np

from pqueens.interfaces import from_config_create_interface

from .model import Model


class SimulationModel(Model):
    """Simulation model class.

    Attributes:
        interface (interface): Interface to simulations/functions.
        response_grad (np.array): Gradient of the model response
    """

    def __init__(self, model_name, interface, **kwargs):
        """Initialize simulation model.

        Args:
            model_name (string):        Name of model
            interface (interface):      Interface to simulator
        """
        super().__init__(model_name)
        self.interface = interface
        self.response_grad = None

    @classmethod
    def from_config_create_model(cls, model_name, config):
        """Create simulation model from problem description.

        Args:
            model_name (string): Name of model
            config (dict):       Dictionary containing problem description

        Returns:
            simulation_model: Instance of SimulationModel
        """
        model_options = config[model_name].copy()
        interface_name = model_options.pop('interface_name')
        interface = from_config_create_interface(interface_name, config)
        model_options.pop('type')

        return cls(model_name=model_name, interface=interface, **model_options)

    def evaluate(self, samples):
        """Evaluate model with current set of input samples.

        Args:
            samples (np.ndarray): Input samples

        Returns:
            self.response (np.array): Response of the underlying model at current variables
        """
        self.response = self.interface.evaluate(samples)
        return self.response

    def grad(self, samples, upstream_gradient):
        """Evaluate gradient of model w.r.t. current set of input samples.

        Args:
            samples (np.array): Input samples
            upstream_gradient (np.array): Upstream gradient function evaluated at input samples
        """
        if self.response.get('gradient') is None:
            raise ValueError('Gradient information not available.')
        # The shape of the returned gradient is weird
        response_gradient = np.swapaxes(self.response['gradient'], 1, 2)
        return np.sum(upstream_gradient[:, :, np.newaxis] * response_gradient, axis=1)
