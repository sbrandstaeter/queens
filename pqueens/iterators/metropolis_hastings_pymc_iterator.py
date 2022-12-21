"""Metropolis Hastings algorithm.

"The Metropolis Hastings algorithm is a not-gradient based MCMC
algortihm. It implements a random walk.
"""

import logging

import numpy as np
import pymc as pm

from pqueens.iterators.pymc_iterator import PyMCIterator
from pqueens.utils.pymc import PymcDistributionWrapperWithoutGrad

_logger = logging.getLogger(__name__)


class MetropolisHastingsPyMCIterator(PyMCIterator):
    """Iterator based on HMC algorithm.

    The Metropolis Hastings sampler is a basic MCMC sampler.

    Attributes:
        covariance (np.array): Covariance for proposal distribution
        tune_interval: frequency of tuning
        scaling (float): Initial scale factor for proposal
    Returns:
        metropolis_hastings_iterator (obj): Instance of Metropolis-Hastings Iterator
    """

    def __init__(
        self,
        global_settings,
        model,
        num_burn_in,
        num_chains,
        num_samples,
        init_strategy,
        discard_tuned_samples,
        result_description,
        seed,
        use_queens_prior,
        progressbar,
        covariance,
        tune_interval,
        scaling,
    ):
        """Initialize Metropolis Hastings iterator.

        Args:
            global_settings (dict): Global settings of the QUEENS simulations
            model (obj): Underlying simulation model on which the inverse analysis is conducted
            num_burn_in (int): Number of burn-in steps
            num_chains (int): Number of chains to sample
            num_samples (int): Number of samples to generate per chain, excluding burn-in period
            init_strategy (str): Strategy to tune mass damping matrix
            discard_tuned_samples (boolean): Setting to discard the samples of the burin-in period
            result_description (dict): Settings for storing and visualizing the results
            seed (int): Seed for rng
            use_queens_prior (boolean): Setting for using the PyMC priors or the QUEENS prior
            functions
            progressbar (boolean): Setting for printing progress bar while sampling
            covariance (np.array): Covariance for proposal distribution
            tune_interval: frequency of tuning
            scaling (float): Initial scale factor for proposal
        Returns:
            Initialise pymc iterator
        """
        super().__init__(
            global_settings,
            model,
            num_burn_in,
            num_chains,
            num_samples,
            init_strategy,
            discard_tuned_samples,
            result_description,
            seed,
            use_queens_prior,
            progressbar,
        )

        self.covariance = covariance
        self.tune_interval = tune_interval
        self.scaling = scaling

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create Metropolis Hastings iterator from problem description.

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator:MetropolisHastingsPyMCIterator object
        """
        _logger.info(
            "Metropolis Hastings (PyMC) Iterator for experiment: {0}".format(
                config.get('global_settings').get('experiment_name')
            )
        )

        method_options = config[iterator_name]['method_options']

        (
            global_settings,
            model,
            num_burn_in,
            num_chains,
            num_samples,
            init_strategy,
            discard_tuned_samples,
            result_description,
            seed,
            use_queens_prior,
            progressbar,
        ) = super().get_base_attributes_from_config(config, iterator_name, model)

        covariance = method_options.get('covariance', None)
        if covariance is not None:
            covariance = np.array(covariance)
        tune_interval = method_options.get('tune_interval', 100)
        scaling = method_options.get('scaling', 1.0)

        return cls(
            global_settings=global_settings,
            model=model,
            num_burn_in=num_burn_in,
            num_chains=num_chains,
            num_samples=num_samples,
            init_strategy=init_strategy,
            discard_tuned_samples=discard_tuned_samples,
            result_description=result_description,
            seed=seed,
            use_queens_prior=use_queens_prior,
            progressbar=progressbar,
            covariance=covariance,
            tune_interval=tune_interval,
            scaling=scaling,
        )

    def eval_log_prior_grad(self, samples):
        """Evaluate the gradient of the log-prior.

        Args:
            samples (np.array): Samples to evaluate the gradient at

        Returns:
            (np.array): Gradients
        """
        raise NotImplementedError("No gradients are needed for Metropolis-Hastings")

    def eval_log_likelihood(self, samples):
        """Evaluate the log-likelihood.

        Args:
             samples (np.array): Samples to evaluate the likelihood at

        Returns:
            (np.array): log-likelihoods
        """
        log_likelihood = self.model.evaluate(samples, gradient_bool=False)
        return log_likelihood

    def eval_log_likelihood_grad(self, samples):
        """Evaluate the gradient of the log-likelihood.

        Args:
            samples (np.array): Samples to evaluate the gradient at

        Returns:
            (np.array): Gradients
        """
        raise NotImplementedError("No gradients are used for Metropolis-Hastings")

    def init_mcmc_method(self):
        """Init the PyMC MCMC Model.

        Args:

        Returns:
            step (obj): The MCMC Method within the PyMC Model
        """
        step = pm.Metropolis(
            S=self.covariance,
            scaling=self.scaling,
            tune_interval=self.tune_interval,
        )
        return step

    def init_distribution_wrapper(self):
        """Init the PyMC wrapper for the QUEENS distributions."""
        self.log_like = PymcDistributionWrapperWithoutGrad(self.eval_log_likelihood)
        if self.use_queens_prior:
            self.log_prior = PymcDistributionWrapperWithoutGrad(self.eval_log_prior)
        _logger.info("Initialize Metropolis Hastings by PyMC run.")
