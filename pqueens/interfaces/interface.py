"""Interface class to map input variables to simulation outputs."""
import abc

import pqueens.parameters.parameters as parameters_module


class Interface(metaclass=abc.ABCMeta):
    """Interface class to map input variables to simulation outputs.

    The interface is responsible for the actual mapping between input
    variables and simulation outputs. The purpose of this base class is
    to define a unified interface on the one hand, while at the other
    hand taking care of the construction of the appropriate objects from
    the derived class.

    Attributes:
        name (str): Name of the interface
        parameters (obj): Parameters object
    """

    def __init__(self, name):
        """Initialize interface object.

        Args:
            name (obj): Name of the interface.
        """
        self.name = name
        self.parameters = parameters_module.parameters

    @abc.abstractmethod
    def evaluate(self, samples, gradient_bool=False):
        """Evaluate samples.

        Mapping function which orchestrates call to external simulation
        software or approximation.

        Args:
            samples (list):  list of variables objects
            gradient_bool (bool): Flag to determine, whether the gradient of the function at
                                  the evaluation point is expected (True) or not (False)
        """
