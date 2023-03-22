"""Base module for iterators or methods."""

import abc

import pqueens.parameters.parameters as parameters_module
from pqueens.models import from_config_create_model


class Iterator(metaclass=abc.ABCMeta):
    """Base class for Iterator hierarchy.

    This Iterator class is the base class for one of the primary class
    hierarchies in QUEENS. The job of the iterator hierarchy is to
    coordinate and execute simulations/function evaluations.

    Attributes:
        model (obj, optional): Model to be evaluated by iterator.
        global_settings (dict, optional): Settings for the QUEENS run.
        parameters: Parameters object
    """

    def __init__(self, model=None, global_settings=None):
        """Initialize iterator object.

        Args:
            model (obj, optional): Model to be evaluated by iterator.
            global_settings (dict, optional): Settings for the QUEENS run.
        """
        self.model = model
        self.global_settings = global_settings
        self.parameters = parameters_module.parameters

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create iterator from problem description.

        Args:
            config (dict):       Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: Iterator object
        """
        method_options = config[iterator_name].copy()
        method_options.pop('type')
        if model is None:
            model_name = method_options['model_name']
            model = from_config_create_model(model_name, config)
        method_options.pop('model_name', None)
        global_settings = config['global_settings']
        return cls(model=model, global_settings=global_settings, **method_options)

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
