"""Base module for iterators or methods."""

import abc

import queens.global_settings


class Iterator(metaclass=abc.ABCMeta):
    """Base class for Iterator hierarchy.

    This Iterator class is the base class for one of the primary class
    hierarchies in QUEENS. The job of the iterator hierarchy is to
    coordinate and execute simulations/function evaluations.

    Attributes:
        model (obj): Model to be evaluated by iterator.
        experiment_name (str): Experiment name
        output_dir (Path): Output directory
        parameters: Parameters object
    """

    def __init__(self, model, parameters):
        """Initialize iterator object.

        Args:
            model (Model): Model to be evaluated by iterator.
            parameters (Parameters): Parameters object
        """
        self.model = model
        self.experiment_name = queens.global_settings.GLOBAL_SETTINGS.experiment_name
        self.output_dir = queens.global_settings.GLOBAL_SETTINGS.output_dir
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
