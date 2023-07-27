"""Latin hypercube sampling iterator."""

import logging

import numpy as np
from pyDOE import lhs

from pqueens.iterators.iterator import Iterator
from pqueens.utils.process_outputs import process_outputs, write_results

_logger = logging.getLogger(__name__)


class LHSIterator(Iterator):
    """Basic LHS Iterator to enable Latin Hypercube sampling.

    Attributes:
        seed (int): Seed for numpy random number generator.
        num_samples (int):    Number of samples to compute.
        num_iterations (int): Number of optimization iterations of design.
        result_description (dict):  Description of desired results.
        criterion (str): Allowable values are:

            *   *center* or *c*
            *   *maximin* or *m*
            *   *centermaximin* or *cm*
            *   *correlation* or *corr*
        samples (np.array):   Array with all samples.
        output (np.array):   Array with all model outputs.
    """

    def __init__(
        self,
        model,
        global_settings,
        parameters,
        seed,
        num_samples,
        result_description=None,
        num_iterations=10,
        criterion='maximin',
    ):
        """Initialise LHSiterator.

        Args:
            model (obj, optional): Model to be evaluated by iterator.
            global_settings (dict): Settings for the QUEENS run.
            parameters (obj): Parameters object
            seed (int): Seed for numpy random number generator
            num_samples (int):    Number of samples to compute
            result_description (dict, opt):  Description of desired results
            num_iterations (int): Number of optimization iterations of design
            criterion (str): Allowable values are "center" or "c", "maximin" or "m",
                             "centermaximin" or "cm", and "correlation" or "corr"
        """
        super().__init__(model, global_settings, parameters)
        self.seed = seed
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.result_description = result_description
        self.criterion = criterion
        self.samples = None
        self.output = None

    def pre_run(self):
        """Generate samples for subsequent LHS analysis."""
        np.random.seed(self.seed)

        num_inputs = self.parameters.num_parameters

        _logger.info('Number of inputs: %s', num_inputs)
        _logger.info('Number of samples: %s', self.num_samples)
        _logger.info('Criterion: %s', self.criterion)
        _logger.info('Number of iterations: %s', self.num_iterations)

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
            results = process_outputs(self.output, self.result_description, input_data=self.samples)
            if self.result_description["write_results"] is True:
                write_results(
                    results,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )

        _logger.info("Size of inputs %s", self.samples.shape)
        _logger.debug("Inputs %s", self.samples)
        _logger.info("Size of outputs %s", self.output['mean'].shape)
        _logger.debug("Outputs %s", self.output['mean'])
