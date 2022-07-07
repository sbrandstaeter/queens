"""Latin hypercube sampling iterator."""

import logging

import numpy as np
from pyDOE import lhs

from pqueens.iterators.iterator import Iterator
from pqueens.models import from_config_create_model
from pqueens.utils.process_outputs import process_ouputs, write_results

_logger = logging.getLogger(__name__)


class LHSIterator(Iterator):
    """Basic LHS Iterator to enable Latin Hypercube sampling.

    Attributes:
        model (model):        Model to be evaluated by iterator
        seed  (int):          Seed for random number generation
        num_samples (int):    Number of samples to compute
        num_iterations (int): Number of optimization iterations of design
        result_description (dict):  Description of desired results
        seed (int): Seed for numpy random number generator
        samples (np.array):   Array with all samples
        output (np.array):   Array with all model outputs
        criterion (str): Allowable values are "center" or "c", "maximin" or "m",
                         "centermaximin" or "cm", and "correlation" or "corr"
    """

    def __init__(
        self,
        model,
        seed,
        num_samples,
        num_iterations,
        result_description,
        global_settings,
        criterion,
    ):
        """Initialise LHSiterator.

        Args:
            model (obj, optional): Model to be evaluated by iterator.
            global_settings (dict, optional): Settings for the QUEENS run.
            num_samples (int):    Number of samples to compute
            num_iterations (int): Number of optimization iterations of design
            result_description (dict):  Description of desired results
            seed (int): Seed for numpy random number generator
            criterion (str): Allowable values are "center" or "c", "maximin" or "m",
                             "centermaximin" or "cm", and "correlation" or "corr"
        """
        super().__init__(model, global_settings)
        self.seed = seed
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.result_description = result_description
        self.criterion = criterion
        self.samples = None
        self.output = None

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create LHS iterator from problem description.

        Args:
            config (dict):       Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: LHSIterator object
        """
        method_options = config[iterator_name]["method_options"]
        if model is None:
            model_name = method_options["model"]
            model = from_config_create_model(model_name, config)

        result_description = method_options.get("result_description", None)
        global_settings = config.get("global_settings", None)

        return cls(
            model,
            method_options["seed"],
            method_options["num_samples"],
            method_options.get("num_iterations", 10),
            result_description,
            global_settings,
            method_options.get("criterion", "maximin"),
        )

    def pre_run(self):
        """Generate samples for subsequent LHS analysis."""
        np.random.seed(self.seed)

        num_inputs = self.parameters.num_parameters

        _logger.info(f'Number of inputs: {num_inputs}')
        _logger.info(f'Number of samples: {self.num_samples}')
        _logger.info(f'Criterion: {self.criterion}')
        _logger.info(f'Number of iterations: {self.num_iterations}')

        # create latin hyper cube samples in unit hyper cube
        hypercube_samples = lhs(
            num_inputs, self.num_samples, criterion=self.criterion, iterations=self.num_iterations
        )
        # scale and transform samples according to the inverse cdf
        self.samples = self.parameters.inverse_cdf_transform(hypercube_samples)

    def core_run(self):
        """Run LHS Analysis on model."""
        self.output = self.model.evaluate(self.samples)

    def post_run(self):
        """Analyze the results."""
        if self.result_description is not None:
            results = process_ouputs(self.output, self.result_description)
            if self.result_description["write_results"] is True:
                write_results(
                    results,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )

        _logger.info("Size of inputs {}".format(self.samples.shape))
        _logger.debug("Inputs {}".format(self.samples))
        _logger.info("Size of outputs {}".format(self.output['mean'].shape))
        _logger.debug("Outputs {}".format(self.output['mean']))
