"""Metropolis-Hastings algorithm.

"The Metropolis-Hastings algorithm is a Markov Chain Monte Carlo (MCMC)
method for obtaining a sequence of random samples from a probability
distribution from which direct sampling is difficult." [1]

References:
    [1]: https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
"""

import logging

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from pqueens.distributions import from_config_create_distribution
from pqueens.distributions.normal import NormalDistribution
from pqueens.iterators.iterator import Iterator
from pqueens.models import from_config_create_model
from pqueens.utils import mcmc_utils, smc_utils
from pqueens.utils.process_outputs import process_ouputs, write_results

_logger = logging.getLogger(__name__)


class MetropolisHastingsIterator(Iterator):
    """Iterator based on Metropolis-Hastings algorithm.

    The Metropolis-Hastings algorithm can be considered the benchmark
    Markov Chain Monte Carlo (MCMC) algorithm. It may be used to sample
    from complex, intractable probability distributions from which
    direct sampling is difficult or impossible.
    The implemented version is a random walk Metropolis-Hastings
    algorithm.

    Attributes:
        as_mcmc_kernel (bool): indicates whether the iterator is used
                               as an MCMC kernel for a SMC iterator or
                               as the main iterator itself
        accepted (int): number of accepted proposals
        log_likelihood (np.array): log of pdf of likelihood at samples
        log_posterior (np.array): log of pdf of posterior at samples
        log_prior (np.array): log of pdf of prior at samples
        num_burn_in (int): Number of burn-in samples
        num_chains (int): Number of independent chains
        num_samples (int): Number of samples per chain
        tot_num_samples (int): Total number of samples per chain, i.e.,
                               (initial + burn-in + chain)
        proposal_distribution (scipy.stats.rv_continuous): Proposal distribution
                                                           giving zero-mean deviates
        result_description (dict):  Description of desired results
        chains (numpy.array): Array with all samples
        scale_covariance (float): Scale of covariance matrix
                                  of gaussian proposal distribution
        seed (int): Seed for random number generator
        tune (bool): Tune the scale of covariance
        tune_interval (int): Tune the scale of the covariance every
                             tune_interval-th step
    """

    def __init__(
        self,
        as_mcmc_kernel,
        global_settings,
        model,
        num_burn_in,
        num_chains,
        num_samples,
        proposal_distribution,
        result_description,
        scale_covariance,
        seed,
        temper_type,
        tune,
        tune_interval,
    ):
        super().__init__(model, global_settings)

        self.num_chains = num_chains
        self.num_samples = num_samples

        self.proposal_distribution = proposal_distribution

        self.result_description = result_description
        self.as_mcmc_kernel = as_mcmc_kernel

        if not self.as_mcmc_kernel:
            # actually this is not completely true: it the generic
            # tempering type might be useful to generate samples from
            # generic distributions that are not interpreted as posteriors
            # in the sense of Bayes rule.
            # To do so one would need to set temper_type = 'generic'
            # and fix tempering_parameter = gamma = 1.0
            if temper_type != 'bayes':
                raise ValueError(
                    "Plain MH can not handle tempering.\n"
                    "Tempering may only be used in conjunction with SMC."
                )

            self.tune = tune
            # TODO change back to a scalar value.
            #  Otherwise the diagnostics tools might not make sense
            self.scale_covariance = np.ones((self.num_chains, 1)) * scale_covariance

            self.num_burn_in = num_burn_in
        else:
            # scaling and tuning of proposal distribution is done within SMC
            self.tune = False
            self.scale_covariance = 1.0
            self.num_burn_in = 0

            if not isinstance(self.proposal_distribution, NormalDistribution):
                raise RuntimeError("Currently only Normal proposals are supported as MCMC Kernel.")

        self.temper = smc_utils.temper_factory(temper_type)
        # fixed within MH, adapted by SMC if needed
        self.gamma = 1.0

        self.tune_interval = tune_interval

        # check agreement of dimensions of proposal distribution and
        # parameter space
        num_parameters = self.parameters.num_parameters
        if num_parameters != self.proposal_distribution.dimension:
            raise ValueError(
                "Dimensions of proposal distribution and parameter space do not agree."
            )

        self.tot_num_samples = self.num_samples + self.num_burn_in + 1
        self.chains = np.zeros((self.tot_num_samples, self.num_chains, num_parameters))
        self.log_likelihood = np.zeros((self.tot_num_samples, self.num_chains, 1))
        self.log_prior = np.zeros((self.tot_num_samples, self.num_chains, 1))
        self.log_posterior = np.zeros((self.tot_num_samples, self.num_chains, 1))

        self.seed = seed

        self.accepted = np.zeros((self.num_chains, 1))
        self.accepted_interval = np.zeros((self.num_chains, 1))

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None, temper_type='bayes'):
        """Create Metropolis-Hastings iterator from problem description.

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: MetropolisHastingsIterator object
        """
        _logger.info(
            "Metropolis-Hastings Iterator for experiment: %s",
            config.get('global_settings').get('experiment_name'),
        )
        method_options = config[iterator_name]['method_options']
        if model is None:
            model_name = method_options['model']
            model = from_config_create_model(model_name, config)

        result_description = method_options.get('result_description', None)
        global_settings = config.get('global_settings', None)

        # initialize proposal distribution
        name_proposal_distribution = method_options['proposal_distribution']
        prop_opts = config.get(name_proposal_distribution, None)
        if prop_opts is not None:
            proposal_distribution = from_config_create_distribution(prop_opts)
        else:
            raise ValueError(
                f'Could not find proposal distributio'
                f' "{name_proposal_distribution}" in input file.'
            )

        tune = method_options.get('tune', False)
        tune_interval = method_options.get('tune_interval', 100)
        scale_cov = method_options.get('scale_covariance', 1.0)

        num_chains = method_options.get('num_chains', 1)

        num_burn_in = method_options.get('num_burn_in', 0)

        as_mcmc_kernel = method_options.get('as_mcmc_kernel', False)

        return cls(
            as_mcmc_kernel=as_mcmc_kernel,
            global_settings=global_settings,
            model=model,
            num_burn_in=num_burn_in,
            num_chains=num_chains,
            num_samples=method_options['num_samples'],
            proposal_distribution=proposal_distribution,
            result_description=result_description,
            scale_covariance=scale_cov,
            seed=method_options['seed'],
            temper_type=temper_type,
            tune=tune,
            tune_interval=tune_interval,
        )

    def eval_log_prior(self, samples):
        """Evaluate natural logarithm of prior at samples of chains.

        Note: we assume a multiplicative split of prior pdf
        """
        return self.parameters.joint_logpdf(samples).reshape(-1, 1)

    def eval_log_likelihood(self, samples):
        """Evaluate natural logarithm of likelihood at samples of chains."""
        log_likelihood = self.model.evaluate(samples)
        return log_likelihood

    def do_mh_step(self, step_id):
        """Metropolis (Hastings) step."""
        # tune covariance of proposal
        if not step_id % self.tune_interval and self.tune:
            accept_rate_interval = np.exp(
                np.log(self.accepted_interval) - np.log(self.tune_interval)
            )
            if not self.as_mcmc_kernel:
                _logger.info("Current acceptance rate: %s.", accept_rate_interval)
            self.scale_covariance = mcmc_utils.tune_scale_covariance(
                self.scale_covariance, accept_rate_interval
            )
            self.accepted_interval = np.zeros((self.num_chains, 1))

        cur_sample = self.chains[step_id - 1]
        # the scaling only holds for random walks
        delta_proposal = (
            self.proposal_distribution.draw(num_draws=self.num_chains) * self.scale_covariance
        )
        proposal = cur_sample + delta_proposal

        log_likelihood_prop = self.eval_log_likelihood(proposal)
        log_prior_prop = self.eval_log_prior(proposal)

        log_posterior_prop = self.temper(log_prior_prop, log_likelihood_prop, self.gamma)
        log_accept_prob = log_posterior_prop - self.log_posterior[step_id - 1]

        new_sample, accepted = mcmc_utils.mh_select(log_accept_prob, cur_sample, proposal)
        self.accepted += accepted
        self.accepted_interval += accepted

        self.chains[step_id] = new_sample

        self.log_likelihood[step_id] = np.where(
            accepted, log_likelihood_prop, self.log_likelihood[step_id - 1]
        )
        self.log_prior[step_id] = np.where(accepted, log_prior_prop, self.log_prior[step_id - 1])
        self.log_posterior[step_id] = np.where(
            accepted, log_posterior_prop, self.log_posterior[step_id - 1]
        )

    def pre_run(
        self,
        initial_samples=None,
        initial_log_like=None,
        initial_log_prior=None,
        gamma=1.0,
        cov_mat=None,
    ):
        """Draw initial sample."""
        if not self.as_mcmc_kernel:
            _logger.info("Initialize Metropolis-Hastings run.")

            np.random.seed(self.seed)

            # draw initial sample from prior distribution
            initial_samples = self.parameters.draw_samples(self.num_chains)
            initial_log_like = self.eval_log_likelihood(initial_samples)
            initial_log_prior = self.eval_log_prior(initial_samples)
        else:
            # create random walk normal proposal
            mean = np.zeros(cov_mat.shape[0])
            dist_opts = {'distribution': 'normal', 'mean': mean, 'covariance': cov_mat}
            self.proposal_distribution = from_config_create_distribution(dist_opts)

        self.gamma = gamma

        self.chains[0] = initial_samples
        self.log_likelihood[0] = initial_log_like
        self.log_prior[0] = initial_log_prior

        self.log_posterior[0] = self.temper(self.log_prior[0], self.log_likelihood[0], self.gamma)

    def core_run(self):
        """Core run of Metropolis-Hastings iterator.

        1.) Burn-in phase
        2.) Sampling phase
        """
        if not self.as_mcmc_kernel:
            _logger.info('Metropolis-Hastings core run.')

        # Burn-in phase
        for i in range(1, self.num_burn_in + 1):
            self.do_mh_step(i)

        if self.num_burn_in:
            burn_in_accept_rate = np.exp(np.log(self.accepted) - np.log(self.num_burn_in))
            _logger.info("Acceptance rate during burn in: %s", burn_in_accept_rate)
        # reset number of accepted samples
        self.accepted = np.zeros((self.num_chains, 1))
        self.accepted_interval = 0

        # Sampling phase
        for i in range(self.num_burn_in + 1, self.num_burn_in + self.num_samples + 1):
            self.do_mh_step(i)

    def post_run(self):
        """Analyze the resulting chain."""
        avg_accept_rate = np.exp(
            np.log(np.sum(self.accepted)) - np.log((self.num_samples * self.num_chains))
        )
        if self.as_mcmc_kernel:
            # the iterator is used as MCMC kernel for the Sequential Monte Carlo iterator
            return [
                self.chains[-1],
                self.log_likelihood[-1],
                self.log_prior[-1],
                self.log_posterior[-1],
                avg_accept_rate,
            ]
        if self.result_description:
            initial_samples = self.chains[0]
            chain_burn_in = self.chains[1 : self.num_burn_in + 1]
            chain_core = self.chains[self.num_burn_in + 1 : self.num_samples + self.num_burn_in + 1]

            accept_rate = np.exp(np.log(self.accepted) - np.log(self.num_samples))

            # process output takes a dict as input with key 'mean'
            results = process_ouputs(
                {
                    'mean': chain_core,
                    'accept_rate': accept_rate,
                    'chain_burn_in': chain_burn_in,
                    'initial_sample': initial_samples,
                    'log_likelihood': self.log_likelihood,
                    'log_prior': self.log_prior,
                    'log_posterior': self.log_posterior,
                },
                self.result_description,
            )
            if self.result_description["write_results"]:
                write_results(
                    results,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )

            _logger.info("Size of outputs %s", chain_core.shape)
            for i in range(self.num_chains):
                _logger.info("#############################################")
                _logger.info("Chain %s", i + 1)
                _logger.info("\tAcceptance rate: %s", accept_rate[i])
                _logger.info(
                    "\tCovariance of proposal : %s",
                    (self.scale_covariance[i] * self.proposal_distribution.covariance).tolist(),
                )
                _logger.info(
                    "\tmean±std: %s±%s",
                    results.get('mean', np.array([np.nan] * self.num_chains))[i],
                    np.sqrt(results.get('var', np.array([np.nan] * self.num_chains))[i]),
                )
                _logger.info(
                    "\tvar: %s", results.get('var', np.array([np.nan] * self.num_chains))[i]
                )
                _logger.info(
                    "\tcov: %s",
                    results.get('cov', np.array([np.nan] * self.num_chains))[i].tolist(),
                )
            _logger.info("#############################################")

            data_dict = {
                variable_name: np.swapaxes(chain_core[:, :, i], 1, 0)
                for i, variable_name in enumerate(self.parameters.parameters_keys)
            }
            inference_data = az.convert_to_inference_data(data_dict)

            rhat = az.rhat(inference_data)
            _logger.info(rhat)
            ess = az.ess(inference_data, relative=True)
            _logger.info(ess)
            az.plot_trace(inference_data)
            filebasename = (
                f"{self.global_settings['output_dir']}/{self.global_settings['experiment_name']}"
            )
            plt.savefig(filebasename + "_trace.png")

            az.plot_autocorr(inference_data)
            plt.savefig(filebasename + "_autocorr.png")
            plt.close("all")

        return None
