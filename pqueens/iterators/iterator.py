import abc


class Iterator(metaclass=abc.ABCMeta):
    """ Base class for Iterator hierarchy

    This Iterator class is the base class for one of the primary class
    hierarchies in QUEENS.The job of the iterator hierarchy is to coordinate
    and execute simulations/function evaluations. The purpose of this base class
    is twofold. First, it defines the unified interface of the iterator hierarchy.
    Second, it works as factory which allows unified instanciation of iterator object
    by calling its classmethods.

    Attributes:
        model (model): Model to be evaluated by iterator

    """

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_config_create_iterator(cls, config):
        """ Create iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description

        Returns:
            iterator: Iterator object

        """
        from .monte_carlo_iterator import MonteCarloIterator
        from .morris_salib_wrapper_iterator import MorrisSALibIterator
        from .saltelli_salib_wrapper_iterator import SaltelliSALibIterator
        from .saltelli_iterator import SaltelliIterator

        method_dict = {'monte_carlo': MonteCarloIterator,
                       'sa_morris_salib': MorrisSALibIterator,
                       'sa_saltelli' : SaltelliIterator,
                       'sa_saltelli_salib' : SaltelliSALibIterator}


        method_name = config['method']['method_name']
        iterator_class = method_dict[method_name]
        return iterator_class.from_config_create_iterator(config)

    def initialize_run(self):
        """ Optional setup step """
        pass

    def pre_run(self):
        """ Optional pre-run portion of run

            Implemented by Iterators which can generate all Variables
            a priori
        """
        pass

    @abc.abstractmethod
    def core_run(self):
        """ Core part of the run, implemented by all derived classes """
        pass

    def post_run(self):
        """ Optional post-run portion of run, e.g., for doing some post processig """
        pass

    def finalize_run(self):
        """ Optional cleanup step """
        pass

    @abc.abstractmethod
    def eval_model(self):
        """ Call the underlying model, implemented by all derived classes  """
        pass

    def run(self):
        """ Orchestrate initialize/pre/core/post/finalize phases """
        self.initialize_run()
        self.pre_run()
        self.core_run()
        self.post_run()
        self.finalize_run()
