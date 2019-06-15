"""
Metropolis-Hastings algorithm

"The Metropolis-Hastings algorithm is a Markov Chain Monte Carlo (MCMC)
method for obtaining a sequence of random samples from a probability
distribution from which direct sampling is difficult." [1]

References:
    [1]: https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm

"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from pqueens.iterators.iterator import Iterator
from pqueens.models.model import Model
from pqueens.utils import mcmc_utils
from pqueens.utils import smc_utils
from pqueens.utils.process_outputs import process_ouputs
from pqueens.utils.process_outputs import write_results


class MetropolisHastingsIterator(Iterator):
    """
    Iterator based on Metropolis-Hastings algorithm

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
        num_chains (int): Number of idependent chains
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

    def __init__(self,
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
                 tune_interval):
        super().__init__(model, global_settings)

        self.num_chains = num_chains
        self.num_burn_in = num_burn_in
        self.num_samples = num_samples

        self.proposal_distribution = proposal_distribution

        self.result_description = result_description
        self.as_mcmc_kernel = as_mcmc_kernel

        if not self.as_mcmc_kernel and temper_type != 'bayes':
            raise ValueError("Plain MH can not handle tempering.\n"
                             "Tempering may only be used in conjuntion with SMC. ")
        # actually this is not completely true: it the generic
        # tempering type might be useful to generate samples from
        # generic distributions that are not interpreted as posteriors
        # in the sense of Bayes rule.
        # To do so one would need to set temper_type = 'generic'
        # and fix tempering_parameter = gamma = 1.0

        self.temper = smc_utils.temper_factory(temper_type)

        # check agreement of dimensions of proposal distribution and
        # parameter space
        num_variables = self.model.variables[0].get_number_of_active_variables()
        if num_variables is not self.proposal_distribution.dimension:
            raise ValueError("Dimensions of proposal distribution"
                             "and parameter space do not agree.")

        self.tot_num_samples = self.num_samples+self.num_burn_in+1
        self.chains = np.zeros((self.tot_num_samples, self.num_chains, num_variables))
        self.log_likelihood = np.zeros((self.tot_num_samples, self.num_chains, 1))
        self.log_prior = np.zeros((self.tot_num_samples, self.num_chains, 1))
        self.log_posterior = np.zeros((self.tot_num_samples, self.num_chains, 1))

        self.tune = tune
        self.tune_interval = tune_interval

        self.scale_covariance = np.ones((self.num_chains, 1)) *scale_covariance

        self.seed = seed

        self.accepted = np.zeros((self.num_chains, 1))
        self.accepted_interval = np.zeros((self.num_chains, 1))

        self.gamma = 1.


    @classmethod
    def from_config_create_iterator(cls,
                                    config,
                                    iterator_name=None,
                                    model=None,
                                    temper_type='bayes'):
        """
        Create Metropolis-Hastings iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: MetropolisHastingsIterator object

        """

        print("Metropolis-Hastings Iterator for experiment: {0}"
              .format(config.get('global_settings').get('experiment_name')))
        if iterator_name is None:
            method_options = config['method']['method_options']
        else:
            method_options = config[iterator_name]['method_options']
        if model is None:
            model_name = method_options['model']
            model = Model.from_config_create_model(model_name, config)

        result_description = method_options.get('result_description', None)
        global_settings = config.get('global_settings', None)

        # initialize proposal distribution
        name_proposal_distribution = method_options['proposal_distribution']
        proposal_options = config.get(name_proposal_distribution, None)
        if proposal_options is not None:
            distribution_type =  proposal_options['type']
            # translate proposal_options to distribution_options of random_variables
            distribution_options = {'distribution': distribution_type,
                                    'distribution_parameter': list([np.squeeze(proposal_options['mean']),
                                                                    np.squeeze(proposal_options['cov'])
                                                                    ]
                                                                   )
                                    }
            proposal_distribution = mcmc_utils.create_proposal_distribution(distribution_options)
        else:
            raise ValueError(f'Could not find proposal distribution'
                             f' "{name_proposal_distribution}" in input file.')

        tune = method_options.get('tune', False)
        tune_interval = method_options.get('tune_interval', 100)

        num_chains = method_options.get('num_chains', 1)

        as_mcmc_kernel = method_options.get('as_mcmc_kernel', False)

        return cls(as_mcmc_kernel=as_mcmc_kernel,
                   global_settings=global_settings,
                   model=model,
                   num_burn_in=method_options['num_burn_in'],
                   num_chains=num_chains,
                   num_samples=method_options['num_samples'],
                   proposal_distribution=proposal_distribution,
                   result_description=result_description,
                   scale_covariance=method_options['scale_covariance'],
                   seed=method_options['seed'],
                   temper_type=temper_type,
                   tune=tune,
                   tune_interval=tune_interval)

    def eval_model(self):
        """ Evaluate model at current samples of chains. """
        result_dict = self.model.evaluate()
        return result_dict

    def eval_log_prior(self, chains):
        """
        Evaluate natural logarithm of prior at samples of chains.

        Note: we assume a multiplicative split of prior pdf
        """

        log_prior = np.zeros((self.num_chains,1))
        for i in range(chains.shape[0]):
            j = 0
            for _, variable in self.model.variables[i].variables.items():
                log_prior[i] += variable['distribution'].logpdf(chains[i, j])
                j += 1

        return log_prior

    def eval_log_likelihood(self, chains):
        """ Evaluate natural logarithm of likelihood at samples of chains. """

        self.model.update_model_from_sample_batch(chains)
        log_likelihood = self.eval_model()['mean']

        return log_likelihood

    def do_mh_step(self, step_id):
        """ Metropolis (Hastings) step. """

        # tune covariance of proposal
        if not step_id % self.tune_interval and self.tune:
            accept_rate_interval = self.accepted_interval / self.tune_interval
            if not self.as_mcmc_kernel:
                print(f"Current acceptance rate: {accept_rate_interval}.")
            self.scale_covariance = mcmc_utils.tune_scale_covariance(self.scale_covariance,
                                                                     accept_rate_interval)
            self.accepted_interval = np.zeros((self.num_chains, 1))

        cur_sample = self.chains[step_id - 1]
        delta_proposal = self.proposal_distribution.draw(num_draws=self.num_chains) * self.scale_covariance
        proposal = cur_sample + delta_proposal

        log_likelihood_prop = self.eval_log_likelihood(proposal)
        log_prior_prop = self.eval_log_prior(proposal)

        log_posterior_prop = self.temper(log_prior_prop, log_likelihood_prop, self.gamma)
        log_accept_prob = log_posterior_prop - self.log_posterior[step_id-1]

        new_sample, accepted = mcmc_utils.mh_select(log_accept_prob, cur_sample, proposal)

        self.accepted += accepted
        self.accepted_interval += accepted

        self.chains[step_id] = new_sample

        self.log_likelihood[step_id] = np.where(accepted, log_likelihood_prop, self.log_likelihood[step_id-1])
        self.log_prior[step_id] = np.where(accepted, log_prior_prop, self.log_prior[step_id-1])
        self.log_posterior[step_id] = np.where(accepted, log_posterior_prop, self.log_posterior[step_id-1])

    def initialize_run(self, initial_samples=None, initial_log_like=None, initial_log_prior=None, gamma=1.0, scale_covariance=1.0):
        """ Draw initial sample. """

        if not self.as_mcmc_kernel:
            print("Initialize Metropolis-Hastings run.")

        # TODO: check conditions (either all are None or none is None)
        if initial_samples is None or initial_log_like is None or initial_log_prior is None:
            np.random.seed(self.seed)

            # draw initial sample from prior distribution
            initial_samples = np.atleast_2d([variable['distribution'].draw(num_draws=self.num_chains)
                                             for model_variable in self.model.variables
                                             for variable_name, variable
                                             in model_variable.variables.items()])
            initial_log_like = self.eval_log_likelihood(initial_samples)
            initial_log_prior = self.eval_log_prior(initial_samples)
            scale_covariance = self.scale_covariance

        self.gamma = gamma
        self.scale_covariance = scale_covariance

        self.chains[0] = initial_samples
        self.log_likelihood[0] = initial_log_like
        self.log_prior[0] = initial_log_prior

        self.log_posterior[0] = self.temper(self.log_prior[0], self.log_likelihood[0], self.gamma)


    def core_run(self):
        """
        Core run of Metropolis-Hastings iterator

        1.) Burn-in phase
        2.) Sampling phase
        """
        if not self.as_mcmc_kernel:
            print('Metropolis-Hastings core run.')

        # Burn-in phase
        for i in range(1, self.num_burn_in + 1):
            self.do_mh_step(i)

        if self.num_burn_in:
            burn_in_accept_rate = self.accepted / self.num_burn_in
            print("Acceptance rate during burn in: {0}".format(burn_in_accept_rate))
        # reset number of accepted samples
        self.accepted = np.zeros((self.num_chains, 1))
        self.accepted_interval = 0

        # Sampling phase
        for i in range(self.num_burn_in + 1, self.num_burn_in + self.num_samples + 1):
            self.do_mh_step(i)

    def post_run(self):
        """ Analyze the resulting chain. """

        avg_accept_rate = np.sum(self.accepted) / (self.num_samples * self.num_chains)
        if self.as_mcmc_kernel:
            # the iterator is used as MCMC kernel for the Sequential Monte Carlo iterator
            return [self.chains[-1],
                    self.log_likelihood[-1],
                    self.log_prior[-1],
                    self.log_posterior[-1],
                    avg_accept_rate]
        elif self.result_description:
            initial_samples= self.chains[0]
            chain_burn_in = self.chains[1: self.num_burn_in + 1]
            chain_core = self.chains[self.num_burn_in + 1:self.num_samples + self.num_burn_in + 1]

            accept_rate = self.accepted / self.num_samples

            # process output takes a dict as input with key 'mean'
            results = process_ouputs({'mean': chain_core,
                                      'accept_rate': accept_rate,
                                      'chain_burn_in': chain_burn_in,
                                      'initial_sample': initial_samples,
                                      'log_likelihood' : self.log_likelihood,
                                      'log_prior' : self.log_prior,
                                      'log_posterior' : self.log_posterior
                                     },
                                     self.result_description)
            if self.result_description["write_results"]:
                write_results(results,
                              self.global_settings["output_dir"],
                              self.global_settings["experiment_name"])

            print("Size of outputs {}".format(chain_core.shape))
            for i in range(self.num_chains):
                print("#############################################")
                print(f"Chain {i+1}")
                print(f"\tAcceptance rate: {accept_rate[i]}")
                print(f"\tCovariance of proposal : {(self.scale_covariance[i] * self.proposal_distribution.covariance).tolist()}")
                print("\tmean±std: {}±{}".format(results.get('mean', np.array([np.nan]*self.num_chains))[i],
                                                 np.sqrt(results.get('var', np.array([np.nan]*self.num_chains))[i])))
                print("\tvar: {}".format(results.get('var', np.array([np.nan]*self.num_chains))[i]))
                print("\tcov: {}".format(results.get('cov', np.array([np.nan]*self.num_chains))[i].tolist()))
            print("#############################################")

            data_dict = { variable_name : np.swapaxes(chain_core[:,:,i], 1, 0) for model_variable in self.model.variables for i, (variable_name, variable) in enumerate(model_variable.variables.items())}
            inference_data = az.convert_to_inference_data(data_dict)

            rhat = az.rhat(inference_data)
            print(rhat)
            ess = az.ess(inference_data, relative=True)
            print(ess)
            az.plot_trace(inference_data)
            plt.savefig(f"{self.global_settings['output_dir']}/{self.global_settings['experiment_name']}_trace.png")
            az.plot_autocorr(inference_data)
            plt.savefig(f"{self.global_settings['output_dir']}/{self.global_settings['experiment_name']}_autocorr.png")
            plt.close("all")
