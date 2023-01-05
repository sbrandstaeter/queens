"""Sobol sequence iterator."""

import logging

from torch.quasirandom import SobolEngine

from pqueens.iterators.iterator import Iterator
from pqueens.models import from_config_create_model
from pqueens.utils.process_outputs import process_ouputs, write_results

_logger = logging.getLogger(__name__)


class SobolSequenceIterator(Iterator):
    """Sobol sequence in multiple dimensions.

    Attributes:
        model (model):        Model to be evaluated by iterator
        number_of_samples (int):    Number of samples to compute
        randomize (bool): Setting this to True will produce scrambled Sobol sequences. Scrambling is
                          capable of producing better Sobol sequences.
        seed  (int): This is the seed for the scrambling. The seed of the random number generator is
                     set to this, if specified. Otherwise, it uses a random seed.
        result_description (dict):  Description of desired results
        samples (np.array):   Array with all samples
        output (np.array):   Array with all model outputs
    """

    def __init__(
        self,
        model,
        seed,
        number_of_samples,
        randomize,
        result_description,
        global_settings,
    ):
        """Initialise Sobol sequence iterator.

        Args:
             model (model): Model to be evaluated by iterator
             number_of_samples (int): Number of samples to compute
             randomize (bool): Setting this to True will produce scrambled Sobol sequences.
                               Scrambling is capable of producing better Sobol sequences.
             seed  (int): This is the seed for the scrambling. The seed of the random number
                          generator is set to this, if specified. Otherwise, it uses a random seed.
             result_description (dict):  Description of desired results
             global_settings (dict, optional): Settings for the QUEENS run.
        """
        super().__init__(model, global_settings)
        self.seed = seed
        self.number_of_samples = number_of_samples
        self.randomize = randomize
        self.result_description = result_description
        self.samples = None
        self.output = None

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create sobol sequence iterator from problem description.

        Args:
            config (dict):       Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: SobolSequenceIterator object
        """
        method_options = config[iterator_name]
        if model is None:
            model_name = method_options["model_name"]
            model = from_config_create_model(model_name, config)

        seed = method_options["seed"]
        number_of_samples = method_options["num_samples"]
        randomize = method_options.get("randomize", False)

        result_description = method_options.get("result_description", None)
        global_settings = config.get("global_settings", None)

        return cls(
            model,
            seed,
            number_of_samples,
            randomize,
            result_description,
            global_settings,
        )

    def pre_run(self):
        """Generate samples for subsequent sobol sequence analysis."""

        _logger.info(f'Number of inputs: {self.parameters.num_parameters}')
        _logger.info(f'Number of samples: {self.number_of_samples}')
        _logger.info(f'Randomize: {self.randomize}')

        # create samples
        sobol_engine = SobolEngine(
            dimension=self.parameters.num_parameters, scramble=self.randomize, seed=self.seed
        )

        qmc_samples = sobol_engine.draw(n=self.number_of_samples)

        # scale and transform samples according to the inverse cdf
        self.samples = self.parameters.inverse_cdf_transform(qmc_samples.numpy().astype('float64'))

    def core_run(self):
        """Run sobol sequence analysis on model."""
        self.output = self.model.evaluate(self.samples)

    def post_run(self):
        """Analyze the results."""
        if self.result_description is not None:
            results = process_ouputs(self.output, self.result_description, input_data=self.samples)
            if self.result_description["write_results"] is True:
                write_results(
                    results,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )
