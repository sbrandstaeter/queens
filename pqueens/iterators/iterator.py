import abc

# TODO add iterator name to attributes
# TODO pull model out of base class and add to all subclasses
# TODO add docstring to global settings


class Iterator(metaclass=abc.ABCMeta):
    """ Base class for Iterator hierarchy

    This Iterator class is the base class for one of the primary class
    hierarchies in QUEENS.The job of the iterator hierarchy is to coordinate
    and execute simulations/function evaluations. The purpose of this base class
    is twofold. First, it defines the unified interface of the iterator hierarchy.
    Second, it works as factory which allows unified instantiation of iterator object
    by calling its classmethods.

    Attributes:
        model (model): Model to be evaluated by iterator

    """

    def __init__(self, model, global_settings):
        self.model = model
        self.global_settings = global_settings

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """ Create iterator from problem description

        Args:
            config (dict):       Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: Iterator object

        """
        from .bayesian_optimization_iterator import BayesOptIterator
        from .data_iterator import DataIterator
        from .lhs_iterator import LHSIterator
        from .lhs_iterator_mf import MF_LHSIterator
        from .metropolis_hastings_iterator import MetropolisHastingsIterator
        from .monte_carlo_iterator import MonteCarloIterator
        from .morris_salib_wrapper_iterator import MorrisSALibIterator
        from pqueens.iterators.optimization_iterator import OptimizationIterator
        from .saltelli_iterator import SaltelliIterator
        from .saltelli_salib_wrapper_iterator import SaltelliSALibIterator
        from .sequential_monte_carlo_iterator import SequentialMonteCarloIterator

        method_dict = {'bayesian_optimization': BayesOptIterator,
                       'lhs': LHSIterator,
                       'lhs_mf': MF_LHSIterator,
                       'metropolis_hastings' : MetropolisHastingsIterator,
                       'monte_carlo': MonteCarloIterator,
                       'optimization' : OptimizationIterator,
                       'read_data_from_file': DataIterator,
                       'sa_morris_salib': MorrisSALibIterator,
                       'sa_saltelli': SaltelliIterator,
                       'sa_saltelli_salib': SaltelliSALibIterator,
                       'smc' : SequentialMonteCarloIterator}

        if iterator_name is None:
            method_name = config['method']['method_name']
            iterator_class = method_dict[method_name]
            iterator = iterator_class.from_config_create_iterator(config, model)
        else:
            method_name = config[iterator_name]['method_name']
            iterator_class = method_dict[method_name]
            iterator = iterator_class.from_config_create_iterator(config, iterator_name, model)
        return iterator

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
