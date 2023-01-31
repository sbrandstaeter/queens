"""Model class."""
import abc

import numpy as np

import pqueens.parameters.parameters as parameters_module


class Model(metaclass=abc.ABCMeta):
    """Base Model class.

        The model hierarchy contains a set of parameters, an interface,
        and a set of responses. An iterator operates on the model to map
        the variables into responses using the interface.

        As with the Iterator hierarchy, the purpose of this base class is
        twofold. One, it defines a unified interface for all derived classes.
        Two, it acts as a factory for the instantiation of model objects.

    Attributes:
        name (str): Name of the model.
        parameters (obj): Parameters object.
        response (np.array): Response corresponding to parameters.
    """

    def __init__(self, name=None):
        """Init model object.

        Args:
            name (optional, string): Name of model
        """
        self.name = name
        self.parameters = parameters_module.parameters
        self.response = None

    @abc.abstractmethod
    def evaluate(self, samples, **kwargs):
        """Evaluate model with current set of samples.

        Args:
            samples (np.array): Current sample batch for which the model response should be
                                calculated.
        """
        pass

    # @abc.abstractmethod
    def _grad(self, samples, upstream):
        """Evaluate gradient of model with current set of samples.

        Args:
            samples (np.array): Evaluated samples
            upstream (np.array): Upstream gradient
        """
        pass

    def evaluate_and_gradient(self, samples):
        model_output = self.evaluate(samples, gradient=True)
        model_gradient = self._grad(samples, np.ones((samples.shape[0], 1)))
        return model_output, model_gradient
