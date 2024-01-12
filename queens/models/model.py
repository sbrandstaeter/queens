"""Model class."""
import abc
from typing import final

import numpy as np


class Model(metaclass=abc.ABCMeta):
    """Base Model class.

        The model hierarchy contains a set of parameters, an interface,
        and a set of responses. An iterator operates on the model to map
        the variables into responses using the interface.

        As with the Iterator hierarchy, the purpose of this base class is
        twofold. One, it defines a unified interface for all derived classes.
        Two, it acts as a factory for the instantiation of model objects.

    Attributes:
        response (dict): Response of the underlying model at input samples.
    """

    evaluate_and_gradient_bool = False

    def __init__(self):
        """Init model object."""
        self.response = None

    @abc.abstractmethod
    def evaluate(self, samples):
        """Evaluate model with current set of input samples.

        Args:
            samples (np.ndarray): Input samples
        """

    @abc.abstractmethod
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

    @final
    def evaluate_and_gradient(self, samples, upstream_gradient=None):
        r"""Evaluate model output and gradient.

        Consider current model f(x) with input samples x, and upstream function g(f). The provided
        upstream gradient is :math:`\frac{\partial g}{\partial f}` and the method returns
        the model output f(x) and the gradient :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`.

        Args:
            samples (np.array): Input samples
            upstream_gradient (np.array, opt): Upstream gradient function evaluated at input samples
                                               :math:`\frac{\partial g}{\partial f}`

        Returns:
            model_output (np.array): Model output
            model_gradient (np.array): Gradient w.r.t. current set of input samples
                                       :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`
        """
        Model.evaluate_and_gradient_bool = True
        model_output = self.evaluate(samples)['result']
        if upstream_gradient is None:
            upstream_gradient = np.ones((samples.shape[0], 1))
        model_gradient = self.grad(
            samples, upstream_gradient=upstream_gradient.reshape(samples.shape[0], 1)
        )
        Model.evaluate_and_gradient_bool = False
        return model_output, model_gradient
