"""Model class."""
import abc
from typing import final

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
        response (dict): Response corresponding to parameters.
    """

    _evaluate_and_gradient_bool = False

    def __init__(self, name=None):
        """Init model object.

        Args:
            name (optional, string): Name of model
        """
        self.name = name
        self.parameters = parameters_module.parameters
        self.response = None

    @abc.abstractmethod
    def evaluate(self, samples):
        """Evaluate model with current set of input samples.

        Args:
            samples (np.ndarray): Input samples
        """

    @abc.abstractmethod
    def grad(self, samples, upstream_gradient):
        """Evaluate gradient of model w.r.t. current set of input samples.

        Args:
            samples (np.array): Input samples
            upstream_gradient (np.array): Upstream gradient function evaluated at input samples
        """

    @final
    def evaluate_and_gradient(self, samples, upstream_gradient=None):
        """Evaluate model output and gradient.

        Args:sam
            samples (np.array): Input samples
            upstream_gradient (np.array, opt): Upstream gradient function evaluated at input samples

        Returns:
            model_output (np.array): Model output
            model_gradient (np.array): Evaluated model gradient w.r.t. the input samples
        """
        Model._evaluate_and_gradient_bool = True
        model_output = self.evaluate(samples)
        if upstream_gradient is None:
            upstream_gradient = np.ones((samples.shape[0], 1))
        model_gradient = self.grad(
            samples, upstream_gradient=upstream_gradient.reshape(samples.shape[0], 1)
        )
        Model._evaluate_and_gradient_bool = False
        return model_output, model_gradient
