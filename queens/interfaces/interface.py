"""Interface class to map input variables to simulation outputs."""

import abc


class Interface(metaclass=abc.ABCMeta):
    """Interface class to map input variables to simulation outputs.

    The interface is responsible for the actual mapping between input
    variables and simulation outputs. The purpose of this base class is
    to define a unified interface on the one hand, while at the other
    hand taking care of the construction of the appropriate objects from
    the derived class.

    Attributes:
        parameters (obj): Parameters object.
        latest_job_id (int):    Latest job ID.
    """

    def __init__(self):
        """Initialize interface object."""
        self.latest_job_id = 0

    @abc.abstractmethod
    def evaluate(self, samples):
        """Evaluate samples.

        Mapping function which orchestrates call to external simulation
        software or approximation.

        Args:
            samples (np.array): Samples of simulation input variables
        """
