"""Base module for iterators or methods."""

import abc


class Iterator(metaclass=abc.ABCMeta):
    """Base class for Iterator hierarchy.

    This Iterator class is the base class for one of the primary class
    hierarchies in QUEENS.The job of the iterator hierarchy is to coordinate
    and execute simulations/function evaluations. The purpose of this base class
    is twofold. First, it defines the unified interface of the iterator hierarchy.
    Second, it works as factory which allows unified instantiation of iterator object
    by calling its classmethods.

    Attributes:
        model (model): Model to be evaluated by iterator
    """

    def __init__(self, model=None, global_settings=None):
        """Initialize base iterator from problem description.

        Args:
            model (obj, optional): Model object on which the iterator
                                   is applied to. Defaults to None.
            global_settings (dict, optional): Dictionary containing global settings
                                              for the QUEENS simulation run.
                                              Defaults to None.
        """
        self.model = model
        self.global_settings = global_settings

    def initialize_run(self):
        """Optional setup step."""
        pass

    def pre_run(self):
        """Optional pre-run portion of run.

        Implemented by Iterators which can generate all Variables a
        priori
        """
        pass

    @abc.abstractmethod
    def core_run(self):
        """Core part of the run, implemented by all derived classes."""
        pass

    def post_run(self):
        """Optional post-run portion of run.

        E.g., for doing some post processing.
        """
        pass

    def finalize_run(self):
        """Optional cleanup step."""
        pass

    @abc.abstractmethod
    def eval_model(self):
        """Call the underlying model, implemented by all derived classes."""
        pass

    def run(self):
        """Orchestrate initialize/pre/core/post/finalize phases."""
        self.initialize_run()
        self.pre_run()
        self.core_run()
        self.post_run()
        self.finalize_run()
