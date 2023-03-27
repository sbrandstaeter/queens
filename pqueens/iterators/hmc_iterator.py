"""HMC algorithm.

"The Hamiltonian Monte Carlo sampler is a gradient based MCMC algortihm.
It is used to sample from arbitrary probability distributions.
"""

import logging

import pymc as pm

from pqueens.iterators.pymc_iterator import PyMCIterator

_logger = logging.getLogger(__name__)


class HMCIterator(PyMCIterator):
    """Iterator based on HMC algorithm.

    The HMC sampler is a state of the art MCMC sampler. It is based on the Hamiltonian mechanics.

    Attributes:
        max_steps (int): Maximum of leapfrog steps to take in one iteration
        target_accept (float): Target accpetance rate which should be conistent after burn-in
        path_length (float): Maximum length of particle trajectory
        step_size (float): Step size, scaled by 1/(parameter dimension **0.25)
        scaling (np.array): The inverse mass, or precision matrix
        is_cov (boolean): Setting if the scaling is a mass or covariance matrix
        init_strategy (str): Strategy to tune mass damping matrix
        advi_iterations (int): Number of iteration steps of ADVI based init strategies

    Returns:
        hmc_iterator (obj): Instance of HMC Iterator
    """

    def __init__(
        self,
        global_settings,
        model,
        num_samples,
        seed,
        num_burn_in=100,
        num_chains=1,
        discard_tuned_samples=True,
        result_description=None,
        summary=True,
        pymc_sampler_stats=False,
        as_inference_dict=False,
        use_queens_prior=False,
        progressbar=False,
        max_steps=100,
        target_accept=0.65,
        path_length=2.0,
        step_size=0.25,
        scaling=None,
        is_cov=False,
        init_strategy='auto',
        advi_iterations=50_000,
    ):
        """Initialize HMC iterator.

        Args:
            global_settings (dict): Global settings of the QUEENS simulations
            model (obj): Underlying simulation model on which the inverse analysis is conducted
            num_samples (int): Number of samples to generate per chain, excluding burn-in period
            seed (int): Seed for rng
            num_burn_in (int, opt): Number of burn-in steps
            num_chains (int, opt): Number of chains to sample
            discard_tuned_samples (boolean, opt): Setting to discard the samples of the burin-in
                                                  period
            result_description (dict, opt): Settings for storing and visualizing the results
            summary (bool, opt):  Print sampler summary
            pymc_sampler_stats (bool, opt): Compute additional sampler statistics
            as_inference_dict (bool, opt): Return inference_data object instead of trace object
            use_queens_prior (boolean, opt): Setting for using the PyMC priors or the QUEENS prior
                                             functions
            progressbar (boolean, opt): Setting for printing progress bar while sampling
            max_steps (int, opt): Maximum of leapfrog steps to take in one iteration
            target_accept (float, opt): Target accpetance rate which should be conistent after
                                        burn-in
            path_length (float, opt): Maximum length of particle trajectory
            step_size (float, opt): Step size, scaled by 1/(parameter dimension **0.25)
            scaling (np.array, opt): The inverse mass, or precision matrix
            is_cov (boolean, opt): Setting if the scaling is a mass or covariance matrix
            init_strategy (str, opt): Strategy to tune mass damping matrix
            advi_iterations (int, opt): Number of iteration steps of ADVI based init strategies
        Returns:
            Initialise pymc iterator
        """
        _logger.info("HMC Iterator for experiment: %s", global_settings.get('experiment_name'))

        super().__init__(
            global_settings,
            model,
            num_burn_in,
            num_chains,
            num_samples,
            discard_tuned_samples,
            result_description,
            summary,
            pymc_sampler_stats,
            as_inference_dict,
            seed,
            use_queens_prior,
            progressbar,
        )
        self.max_steps = max_steps
        self.target_accept = target_accept
        self.path_length = path_length
        self.step_size = step_size
        self.scaling = scaling
        self.is_cov = is_cov
        self.init_strategy = init_strategy
        self.advi_iterations = advi_iterations

    def init_mcmc_method(self):
        """Init the PyMC MCMC Model.

        Args:

        Returns:
            step (obj): The MCMC Method within the PyMC Model
        """
        # have only scaling or potential as mass matrix
        potential = None
        if self.scaling is None:
            # use NUTS init to get potential for the init
            _logger.info("Using NUTS initialization to init HMC, ignore next line.")
            self.initvals, step_helper = pm.init_nuts(
                init=self.init_strategy,
                chains=1,
                initvals=self.initvals,
                progressbar=self.progressbar,
                n_init=self.advi_iterations,
                random_seed=self.seed,
                model=self.pymc_model,
            )
            potential = step_helper.potential
        step = pm.HamiltonianMC(
            target_accept=self.target_accept,
            max_steps=self.max_steps,
            path_length=self.path_length,
            step_scale=self.step_size,
            scaling=self.scaling,
            is_cov=self.is_cov,
            potential=potential,
            model=self.pymc_model,
        )
        return step
