import abc
import numpy as np
import GPy

class Bmfmc(metaclass=abc.ABCMeta):
    """ Base class for Bayesian Multi-Fidelity Monte Carlo algorithms

        The BMFMC abstract class is an interface for several implementations of
        Bayesian Multi-Fidelity Monte Carlo methods. This interface is supposed
        to enable the use of several low- or middle fidelity models that can be
        used sequentially or in parallel.

        Following the specification given in the input file, different random
        processes can be used for the mapping between the fidelity levels.

        Another configuration given by the input file is the number and the
        level of fidelity levels.

    Attributes:
        blabla

    Outputs:
        Vector of QoI calculated on different fidelity levels
    """

    def __init__(self, name, config_para):
        """ Init Bmfmc object

        Args:
            name (string):                      Name of the Bmfmc mapping
            config_para (dict):                 Dictionary with configuration
                                                parameters for the mapping routine
        """
        self.name = name
        self.config_para = config_para

    @classmethod
    def from_config_create_bmfmc(cls):
        bmfmc_dict = {'gp_se_mapping': GaussianProcessMap}
        bmfmc_options = config_para[name]
        bmfmc_class = bmfmc_dict[bmfmc_options["type"]]
        return bmfmc_class.from_config_create_bmfmc(name, config_para)

    @abc.abstractmethod
    def .....(self):
        pass


