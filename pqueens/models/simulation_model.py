"""Simulation model class."""
import numpy as np

from pqueens.interfaces import from_config_create_interface

from .model import Model


class SimulationModel(Model):
    """Simulation model class.

    Attributes:
        interface (interface): Interface to simulations/functions.
    """

    def __init__(self, model_name, interface, **kwargs):
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
        r"""Evaluate gradient of model w.r.t. current set of input samples.

        Consider current model f(x) with input samples x, and upstream function g(f). The provided
        upstream gradient is :math:`\frac{\partial g}{\partial f}` and the method returns
        :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`.

        Args:
            samples (np.array): Input samples
            upstream_gradient (np.array): Upstream gradient function evaluated at input samples
                                          :math:`\frac{\partial g}{\partial f}`

        Returns:
            gradient (np.array): Gradient w.r.t. current set of input samples
                                 :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`
        """
        if self.response.get('gradient') is None:
            raise ValueError('Gradient information not available.')
        # The shape of the returned gradient is weird
        response_gradient = np.swapaxes(self.response['gradient'], 1, 2)
        gradient = np.sum(upstream_gradient[:, :, np.newaxis] * response_gradient, axis=1)
        return gradient
