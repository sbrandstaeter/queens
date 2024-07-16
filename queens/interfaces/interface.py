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

    def __init__(self, parameters):
        """Initialize interface object.

        Args:
            parameters (obj): Parameters object
        """
        self.parameters = parameters
        self.latest_job_id = 0

    @abc.abstractmethod
    def evaluate(self, samples):
        """Evaluate samples.

        Mapping function which orchestrates call to external simulation
        software or approximation.

        Args:
            samples (list):  List of variables objects
        """

    def create_samples_list(self, samples, add_job_id=True):
        """Create a list of sample dictionaries with job id.

        Args:
            samples (np.array): Samples of simulation input variables
            add_job_id (bool): add the job_id to the samples if desired.

        Returns:
            samples_list (list): List of dicts containing samples and job ids
        """
        samples_list = []
        for sample in samples:
            self.latest_job_id += 1
            sample_dict = self.parameters.sample_as_dict(sample)
            if add_job_id:
                sample_dict["job_id"] = self.latest_job_id
            samples_list.append(sample_dict)

        return samples_list
