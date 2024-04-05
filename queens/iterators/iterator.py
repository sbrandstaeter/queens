"""Base module for iterators or methods."""

import abc


class Iterator(metaclass=abc.ABCMeta):
    """Base class for Iterator hierarchy.

    This Iterator class is the base class for one of the primary class
    hierarchies in QUEENS. The job of the iterator hierarchy is to
    coordinate and execute simulations/function evaluations.

    Attributes:
        model (obj): Model to be evaluated by iterator.
        parameters: Parameters object
        global_settings (GlobalSettings): settings of the QUEENS experiment including its name and
                                          the output directory
        experiment_name (str): Experiment name
        output_dir (Path): Output directory
    """

    def __init__(self, model, parameters, global_settings):
        """Initialize iterator object.

        Args:
            model (Model): Model to be evaluated by iterator.
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
        """
        self.model = model
        self.global_settings = global_settings
        self.experiment_name = global_settings.experiment_name
        self.output_dir = global_settings.output_dir
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
