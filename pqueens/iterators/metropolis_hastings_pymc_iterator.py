"""Metropolis Hastings algorithm.

"The Metropolis Hastings algorithm is a not-gradient based MCMC
algortihm. It implements a random walk.
"""

import logging

import numpy as np
import pymc as pm

from pqueens.iterators.pymc_iterator import PyMCIterator
from pqueens.utils.pymc import PymcDistributionWrapper

_logger = logging.getLogger(__name__)


class MetropolisHastingsPyMCIterator(PyMCIterator):
    """Iterator based on HMC algorithm.

    The Metropolis Hastings sampler is a basic MCMC sampler.

    Attributes:
        covariance (np.array): Covariance for proposal distribution
        tune_interval: frequency of tuning
        scaling (float): Initial scale factor for proposal
        seen_samples (list): The 5 most recent sample batches which were used to evaluate the
            likelihood function
        seen_likelihoods (list): The 5 most recent results of the likelihood
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
            discard_tuned_samples,
            result_description,
            seed,
            use_queens_prior,
            progressbar,
        )

        self.covariance = covariance
        self.tune_interval = tune_interval
        self.scaling = scaling

        self.seen_samples = None
        self.seen_likelihoods = None

        if not use_queens_prior and len(self.parameters.to_list()) > 1:
            _logger.warning(
                "PyMC does element wise updates if multiple PymC priors are used, "
                "using QUEENS prior instead."
            )
            self.use_queens_prior = True

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
            "PyMC Metropolis-Hastings Iterator for experiment: %s",
            config.get('global_settings').get('experiment_name'),
        )

        (
            global_settings,
            model,
            num_burn_in,
            num_chains,
            num_samples,
            discard_tuned_samples,
            result_description,
            seed,
            use_queens_prior,
            progressbar,
        ) = super().get_base_attributes_from_config(config, iterator_name, model)

        method_options = config[iterator_name]
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
        log_likelihood = np.zeros(shape=(samples.shape[0]))
        # check if sample was seen in previous acceptance step
        if self.seen_samples is None:
            self.seen_samples = [samples.copy(), samples.copy(), samples.copy()]
            log_likelihood = self.model.evaluate(samples, gradient_bool=False)
            self.seen_likelihoods = [
                log_likelihood.copy(),
                log_likelihood.copy(),
                log_likelihood.copy(),
            ]
        else:
            if np.array_equal(self.seen_samples[0], samples):
                log_likelihood = self.seen_likelihoods[0]
            elif np.array_equal(self.seen_samples[1], samples):
                log_likelihood = self.seen_likelihoods[1]
            else:
                log_likelihood = self.model.evaluate(samples, gradient_bool=False)

        # update list of last samples and likelihoods
        self.seen_samples.pop(0)
        self.seen_samples.append(samples.copy())
        self.seen_likelihoods.pop(0)
        self.seen_likelihoods.append(log_likelihood.copy())

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
        dims = self.num_chains * self.parameters.num_parameters
        if self.covariance is None:
            covariance = np.eye(dims)
        elif self.covariance.shape == (dims, dims):
            covariance = self.covariance
        elif self.covariance.shape == (
            self.parameters.num_parameters,
            self.parameters.num_parameters,
        ):
            covariance = np.kron(np.eye(self.num_chains), self.covariance)
        else:
            raise ValueError("Covariance Matrix has not the right shape.")

        step = pm.Metropolis(
            S=covariance,
            scaling=self.scaling,
            tune_interval=self.tune_interval,
        )
        step.elemwise_update = False
        step.accept_rate_iter = np.zeros(self.num_chains, dtype=float)
        step.accepted_iter = np.zeros(self.num_chains, dtype=bool)
        step.accepted_sum = np.zeros(self.num_chains, dtype=int)

        return step

    def init_distribution_wrapper(self):
        """Init the PyMC wrapper for the QUEENS distributions."""
        self.log_like = PymcDistributionWrapper(self.eval_log_likelihood)
        if self.use_queens_prior:
            self.log_prior = PymcDistributionWrapper(self.eval_log_prior)
        _logger.info("Initialize Metropolis Hastings by PyMC run.")
