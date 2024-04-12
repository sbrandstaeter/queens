"""Base module for iterators or methods."""

import abc

from queens.global_settings import GlobalSettings
from queens.models.model import Model
from queens.parameters import Parameters


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
    """

    def __init__(self, model: Model, parameters: Parameters, global_settings: GlobalSettings):
        """Initialize iterator object.

        Args:
            model (Model): Model to be evaluated by iterator.
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
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
