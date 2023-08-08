"""Monte Carlo iterator."""

import logging

import matplotlib.pyplot as plt
import numpy as np

from queens.iterators.iterator import Iterator
from queens.utils.process_outputs import process_outputs, write_results
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class MonteCarloIterator(Iterator):
    """Basic Monte Carlo Iterator to enable MC sampling.

    Attributes:
        seed  (int): Seed for random number generation.
        num_samples (int): Number of samples to compute.
        result_description (dict):  Description of desired results.
        samples (np.array):         Array with all samples.
        output (np.array):          Array with all model outputs.
    """

    @log_init_args(_logger)
    def __init__(
        self,
        model,
        parameters,
        seed,
        num_samples,
        result_description=None,
    ):
        """Initialise Monte Carlo iterator.

        Args:
            model (model):              Model to be evaluated by iterator
            parameters (obj): Parameters object
            seed  (int):                Seed for random number generation
            num_samples (int):          Number of samples to compute
            result_description (dict, opt):  Description of desired results
        """
        super().__init__(model, parameters)
        self.seed = seed
        self.num_samples = num_samples
        self.result_description = result_description
        self.samples = None
        self.output = None

    def pre_run(self):
        """Generate samples for subsequent MC analysis and update model."""
        np.random.seed(self.seed)
        self.samples = self.parameters.draw_samples(self.num_samples)

    def core_run(self):
        """Run Monte Carlo Analysis on model."""
        self.output = self.model.evaluate(self.samples)

    def post_run(self):
        """Analyze the results."""
        if self.result_description is not None:
            results = process_outputs(self.output, self.result_description, self.samples)
            if self.result_description["write_results"] is True:
                write_results(results, self.output_dir, self.experiment_name)

                # ----------------------------- WIP PLOT OPTIONS ----------------------------
                if self.result_description['plot_results'] is True:
                    # Check for dimensionality of the results
                    plt.rcParams["mathtext.fontset"] = "cm"
                    plt.rcParams.update({'font.size': 23})
                    _, ax = plt.subplots()

                    if results['raw_output_data']['mean'][0].shape[0] > 1:
                        for ele in results['raw_output_data']['mean']:
                            ax.plot(ele[:, 0], ele[:, 1])

                        ax.set_xlabel(r't [s]')
                        ax.set_ylabel(r'$C_L(t)$')
                        plt.show()
                    else:
                        data = results['raw_output_data']['mean']
                        ax.hist(data, bins=200)
                        ax.set_xlabel(r'Count [-]')
                        ax.set_xlabel(r'$C_L(t)$')
                        plt.show()
        # else:
        _logger.debug("Size of inputs %s", self.samples.shape)
        _logger.debug("Inputs %s", self.samples)
        _logger.debug("Size of outputs %s", self.output['mean'].shape)
        _logger.debug("Outputs %s", self.output['mean'])
