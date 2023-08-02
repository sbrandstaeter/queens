"""Base module for iterators or methods."""

import abc


class Iterator(metaclass=abc.ABCMeta):
    """Base class for Iterator hierarchy.

    This Iterator class is the base class for one of the primary class
    hierarchies in QUEENS. The job of the iterator hierarchy is to
    coordinate and execute simulations/function evaluations.

    Attributes:
        model (obj): Model to be evaluated by iterator.
        global_settings (dict): Settings for the QUEENS run.
        parameters: Parameters object
    """

    def __init__(self, model, global_settings, parameters):
        """Initialize iterator object.

        Args:
            model (obj): Model to be evaluated by iterator.
            global_settings (dict): Settings for the QUEENS run.
            parameters (obj): Parameters object
        """
        self.model = model
        self.global_settings = global_settings
        self.parameters = parameters

    def pre_run(self):
        """Optional pre-run portion of run."""

    @abc.abstractmethod
    def core_run(self):
        """Core part of the run, implemented by all derived classes."""

    def post_run(self):
        """Optional post-run portion of run.

        E.g. for doing some post processing.
        """

    def run(self):
        """Orchestrate pre/core/post phases."""
        self.pre_run()
        self.core_run()
        self.post_run()
