"""HMC algorithm.

"The Hamiltonian Monte Carlo sampler is a gradient based MCMC algortihm.
It is used to sample from arbitrary probability distributions.
"""

import logging

import numpy as np
import pymc as pm

from pqueens.iterators.pymc_iterator import PyMCIterator
from pqueens.utils.pymc import PymcDistributionWrapper

_logger = logging.getLogger(__name__)


class HMCIterator(PyMCIterator):
    """Iterator based on HMC algorithm.

    The HMC sampler is a state of the art MCMC sampler. It is based on the Hamiltonian mechanics.

    Attributes:
        max_steps (int): Maximum of leapfrog steps to take in one iteration
        target_accept (float): Target accpetance rate which should be conistent after burn-in
        path_length (float): Maximum length of particle trajectory
        scaling (np.array): The inverse mass, or precision matrix
        is_cov (boolean): Setting if the scaling is a mass or covariance matrix
        current_samples (np.array): Most recent evalutated sample by the likelihood function
        current_gradients (np.array): Gradient of the most recent evaluated sample
    Returns:
        hmc_iterator (obj): Instance of HMC Iterator
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
        max_steps,
        target_accept,
        path_length,
        scaling,
        is_cov,
    ):
        """Initialize HMC iterator.

        Args:
            global_settings (dict): Global settings of the QUEENS simulations
            model (obj): Underlying simulation model on which the inverse analysis is conducted
            num_burn_in (int): Number of burn-in steps
            num_chains (int): Number of chains to sample
            num_samples (int): Number of samples to generate per chain, excluding burn-in period
            discard_tuned_samples (boolean): Setting to discard the samples of the burin-in period
            result_description (dict): Settings for storing and visualizing the results
            seed (int): Seed for rng
            use_queens_prior (boolean): Setting for using the PyMC
                                        priors or the QUEENS prior functions
            progressbar (boolean): Setting for printing progress bar while sampling
            max_steps (int): Maximum of leapfrog steps to take in one iteration
            target_accept (float): Target accpetance rate which should be conistent after burn-in
            path_length (float): Maximum length of particle trajectory
            scaling (np.array): The inverse mass, or precision matrix
            is_cov (boolean): Setting if the scaling is a mass or covariance matrix
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
        if self.num_chains > 1:
            _logger.warning(
                "More than 1 chain not supported, use only 1 chain to get valid samples!"
            )
        self.max_steps = max_steps
        self.target_accept = target_accept
        self.path_length = path_length
        self.scaling = scaling
        self.is_cov = is_cov
        self.current_samples = np.zeros((self.num_chains, self.parameters.num_parameters))
        self.current_gradients = np.zeros((self.num_chains, self.parameters.num_parameters))

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create HMC iterator from problem description.

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator:HMCIterator object
        """
        _logger.info(
            "HMC Iterator for experiment: {0}".format(
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
            discard_tuned_samples,
            result_description,
            seed,
            use_queens_prior,
            progressbar,
        ) = super().get_base_attributes_from_config(config, iterator_name, model)

        max_steps = method_options.get('max_steps', 100)
        target_accept = method_options.get('target_accept', 0.65)
        path_length = method_options.get('path_length', 2.0)
        scaling = method_options.get('scaling', None)
        is_cov = method_options.get('is_cov', False)

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
            max_steps=max_steps,
            target_accept=target_accept,
            path_length=path_length,
            scaling=scaling,
            is_cov=is_cov,
        )

    def eval_log_prior_grad(self, samples):
        """Evaluate the gradient of the log-prior.

        Args:
            samples (np.array): Samples to evaluate the gradient at

        Returns:
            (np.array): Gradients
        """
        return self.parameters.grad_joint_logpdf(samples)

    def eval_log_likelihood(self, samples):
        """Evaluate the log-likelihood.

        Args:
             samples (np.array): Samples to evaluate the likelihood at

        Returns:
            (np.array): log-likelihoods
        """
        self.current_samples = samples

        log_likelihood, gradient = self.model.evaluate(samples, gradient_bool=True)

        self.current_gradients = gradient
        return log_likelihood

    def eval_log_likelihood_grad(self, samples):
        """Evaluate the gradient of the log-likelihood.

        Args:
            samples (np.array): Samples to evaluate the gradient at

        Returns:
            (np.array): Gradients
        """
        # pylint: disable-next=fixme
        # TODO: find better way to do this evaluation

        if np.all(self.current_samples == samples):
            gradient = self.current_gradients
        else:
            _, gradient = self.model.evaluate(samples, gradient_bool=True)
        return gradient

    def init_mcmc_method(self):
        """Init the PyMC MCMC Model.

        Args:

        Returns:
            step (obj): The MCMC Method within the PyMC Model
        """
        step = pm.HamiltonianMC(
            target_accept=self.target_accept,
            max_steps=self.max_steps,
            path_length=self.path_length,
            scaling=self.scaling,
            is_cov=self.is_cov,
        )
        return step

    def init_distribution_wrapper(self):
        """Init the PyMC wrapper for the QUEENS distributions."""
        self.log_like = PymcDistributionWrapper(
            self.eval_log_likelihood, self.eval_log_likelihood_grad
        )
        if self.use_queens_prior:
            self.log_prior = PymcDistributionWrapper(self.eval_log_prior, self.eval_log_prior_grad)
        _logger.info("Initialize HMC by PyMC run.")
