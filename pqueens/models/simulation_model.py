"""Simulation model class."""


from pqueens.interfaces import from_config_create_interface
from pqueens.utils.gradient_handler import from_config_create_grad_handler

from .model import Model


class SimulationModel(Model):
    """Simulation model class.

    Attributes:
        interface (interface obj): Interface to simulations/functions
        gradient_interface (interface obj): Interface for gradient computation
        grad_handler(optional, obj): Gradient handling object that contains implementations
                                        for different schemes of model gradient calculations.
        gradient_response (np.array): Gradient of the model w.r.t. the input samples evaluated
                                       at the input samples
    """

    def __init__(
        self,
        model_name,
        interface,
        grad_handler=None,
    ):
        """Initialize simulation model.

        Args:
            model_name (string): Name of model
            interface (interface obj): Interface to simulator
            gradient_interface (optional, interface obj): Interface for gradient computation
            grad_handler(optional, obj): Gradient handling object that contains implementations
                                         for different schemes of model gradient calculations.
        """
        super().__init__(model_name)
        self.interface = interface
        self.grad_handler = grad_handler
        self.gradient_response = None

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

        # get the correct gradient method
        grad_handler_name = model_options.get("gradient_handler_name")
        if grad_handler_name:
            grad_handler = from_config_create_grad_handler(grad_handler_name, interface, config)
        else:
            grad_handler = None

        return cls(model_name, interface, grad_handler)

    def evaluate(self, samples):
        """Evaluate model at sample points.

        Args:
            samples (np.ndarray): Evaluated samples

        Returns:
            self.response (dict): Response of the underlying model at current variables
        """
        self.response = self.interface.evaluate(samples)
        return self.response

    def evaluate_and_gradient(self, samples, upstream_gradient_fun=None):
        """Evaluate model and the model gradient with current set of samples.

        Args:
            samples (np.ndarray): Evaluated samples
            upstream_gradient_fun (obj): The gradient an upstream objective function w.r.t. the
                                         model output.

        Returns:
            self.response (np.array): Response of the underlying model at current variables
            self.gradient_response (np.array): Gradient response of the underlying model at
                                               current input samples
        """
        if self.grad_handler:
            self.response, self.gradient_response = self.grad_handler.evaluate_and_gradient(
                samples=samples,
                evaluate_fun=self.evaluate,
                upstream_gradient_fun=upstream_gradient_fun,
            )
        else:
            raise AttributeError(
                "You need to specify and provide a `gradient handler` object for the "
                "`simulation_model`, if you want to use gradient functionality!"
            )

        return self.response, self.gradient_response
