"""Monte Carlo iterator."""

import logging

import matplotlib.pyplot as plt
import numpy as np

import pqueens.database.database as DB_module
from pqueens.models import from_config_create_model
from pqueens.utils.process_outputs import process_ouputs, write_results

from .iterator import Iterator

_logger = logging.getLogger(__name__)


class MonteCarloIterator(Iterator):
    """Basic Monte Carlo Iterator to enable MC sampling.

    Attributes:
        model (model):              Model to be evaluated by iterator
        seed  (int):                Seed for random number generation
        num_samples (int):          Number of samples to compute
        result_description (dict):  Description of desired results
        samples (np.array):         Array with all samples
        output (np.array):          Array with all model outputs
        db (obj):                   Data base object
    """

    def __init__(
        self,
        model,
        seed,
        num_samples,
        result_description,
        global_settings,
        db,
    ):
        """Initialise Monte Carlo iterator.

        Args:
            model (model):              Model to be evaluated by iterator
            seed  (int):                Seed for random number generation
            num_samples (int):          Number of samples to compute
            result_description (dict):  Description of desired results
            db (obj):                   Data base object
            global_settings (dict, optional): Settings for the QUEENS run.
        """
        super().__init__(model, global_settings)
        self.seed = seed
        self.num_samples = num_samples
        self.result_description = result_description
        self.samples = None
        self.output = None
        self.db = db

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create MC iterator from problem description.

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: MonteCarloIterator object
        """
        _logger.info(config.get('experiment_name'))
        method_options = config[iterator_name]['method_options']
        if model is None:
            model_name = method_options['model']
            model = from_config_create_model(model_name, config)

        result_description = method_options.get('result_description', None)
        global_settings = config.get('global_settings', None)

        db = DB_module.database

        return cls(
            model,
            method_options['seed'],
            method_options['num_samples'],
            result_description,
            global_settings,
            db,
        )

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
            results = process_ouputs(self.output, self.result_description, self.samples)
            if self.result_description["write_results"] is True:
                write_results(
                    results,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )

                # ------------------------------ WIP PLOT OPTIONS -----------------------------
                if self.result_description['plot_results'] is True:
                    # Check for dimensionality of the results
                    plt.rcParams["mathtext.fontset"] = "cm"
                    plt.rcParams.update({'font.size': 23})
                    fig, ax = plt.subplots()

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
