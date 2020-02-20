import abc


class Interface(metaclass=abc.ABCMeta):
    """
    Interface class to map input variables to simulation outputs

    The interface is responsible for the actual mapping between input variables
    and simulation outputs. The purpose of this base class is to define a unified
    interface on the one hand, while at the other hand taking care of the contruction
    of the appropirate objects from the derived class.

    """

    @classmethod
    def from_config_create_interface(cls, interface_name, config):
        """
        Args:
            interface_name (str):   name of the interface
            config (dict):          dictionary with problem description

        Returns:
            interface:              Instance of one of the derived interface classes
        """

        # import here to avoid issues with circular inclusion
        import pqueens.interfaces.job_interface
        import pqueens.interfaces.direct_python_interface
        import pqueens.interfaces.approximation_interface
        import pqueens.interfaces.approximation_interface_mf
        import pqueens.interfaces.bmfmc_interface

        # pylint: disable=line-too-long
        interface_dict = {
            'job_interface': pqueens.interfaces.job_interface.JobInterface,
            'direct_python_interface': pqueens.interfaces.direct_python_interface.DirectPythonInterface,
            'approximation_interface': pqueens.interfaces.approximation_interface.ApproximationInterface,
            'approximation_interface_mf': pqueens.interfaces.approximation_interface_mf.ApproximationInterfaceMF,
            'bmfmc_interface': pqueens.interfaces.bmfmc_interface.BmfmcInterface,
        }
        # pylint: enable=line-too-long

        interface_options = config[interface_name]
        # determine which object to create
        interface_class = interface_dict[interface_options["type"]]
        return interface_class.from_config_create_interface(interface_name, config)

    @abc.abstractmethod
    def map(self, samples):
        """
        Mapping function which orchestrates call to external simulation software
        or approximation

        Args:
            samples (list):  list of variables objects

        Returns:
            dict:           Dictionary with results. Mean, variance and posterior
                            samples can be contained in this dict
        """
        pass
