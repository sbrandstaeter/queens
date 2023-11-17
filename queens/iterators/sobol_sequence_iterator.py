"""Sobol sequence iterator."""

import logging

from scipy.stats.qmc import Sobol

from queens.iterators.iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import process_outputs, write_results

_logger = logging.getLogger(__name__)


class SobolSequenceIterator(Iterator):
    """Sobol sequence in multiple dimensions.

    Attributes:
        seed  (int): This is the seed for the scrambling. The seed of the random number generator is
                     set to this, if specified. Otherwise, it uses a random seed.
        number_of_samples (int): Number of samples to compute.
        randomize (bool): Setting this to *True* will produce scrambled Sobol sequences. Scrambling
                          is capable of producing better Sobol sequences.
        result_description (dict):  Description of desired results.
        samples (np.array):   Array with all samples.
        output (np.array):   Array with all model outputs.
    """

    @log_init_args(_logger)
    def __init__(
        self,
        model,
        parameters,
        seed,
        number_of_samples,
        result_description,
        randomize=False,
    ):
        """Initialize Sobol sequence iterator.

        Args:
             model (model): Model to be evaluated by iterator
             parameters (obj): Parameters object
             seed  (int): This is the seed for the scrambling. The seed of the random number
                          generator is set to this, if specified. Otherwise, it uses a random seed.
             number_of_samples (int): Number of samples to compute
             result_description (dict):  Description of desired results
             randomize (bool): Setting this to True will produce scrambled Sobol sequences.
                               Scrambling is capable of producing better Sobol sequences.
        """
        super().__init__(model, parameters)

        self.seed = seed
        self.number_of_samples = number_of_samples
        self.randomize = randomize
        self.result_description = result_description
        self.samples = None
        self.output = None

    def pre_run(self):
        """Generate samples for subsequent Sobol sequence analysis."""
        _logger.info('Number of inputs: %s', self.parameters.num_parameters)
        _logger.info('Number of samples: %s', self.number_of_samples)
        _logger.info('Randomize: %s', self.randomize)

        # create samples
        sobol_engine = Sobol(
            d=self.parameters.num_parameters, scramble=self.randomize, seed=self.seed
        )

        qmc_samples = sobol_engine.random(n=self.number_of_samples)
        # scale and transform samples according to the inverse cdf
        self.samples = self.parameters.inverse_cdf_transform(qmc_samples)

    def core_run(self):
        """Run Sobol sequence analysis on model."""
        self.output = self.model.evaluate(self.samples)

    def post_run(self):
        """Analyze the results."""
        if self.result_description is not None:
            results = process_outputs(self.output, self.result_description, input_data=self.samples)
            if self.result_description["write_results"] is True:
                write_results(results, self.output_dir, self.experiment_name)
