"""Model class."""
import abc

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
        """Evaluate model with current set of samples.

        Args:
            samples (np.array): Current sample batch for which the model response should be
                                calculated.
        """
        pass

    @abc.abstractmethod
    def evaluate_and_gradient(self, samples, upstream_gradient_fun):
        """Evaluate model and the model gradient with current set of samples.

        Args:
            samples (np.array): Current sample batch for which the model response should be
                                calculated.
            upstream_gradient_fun (obj): The gradient an objective function w.r.t. the model output.
                                         This function is needed for `adjoint`-based gradient
                                         calculation.
        """
        pass
