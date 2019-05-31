"""
Metropolis-Hastings algorithm

"The Metropolis-Hastings algorithm is a Markov Chain Monte Carlo (MCMC)
method for obtaining a sequence of random samples from a probability
distribution from which direct sampling is difficult." [1]

References:
    [1]: https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm

"""

import numpy as np

from pqueens.iterators.iterator import Iterator
from pqueens.models.model import Model
from pqueens.utils import mcmc_utils
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
        accepted (int): number of accepted proposals
        log_likelihood (np.array): log of pdf of likelihood at samples
        log_posterior (np.array): log of pdf of posterior at samples
        log_prior (np.array): log of pdf of prior at samples
        num_burn_in (int): Number of burn-in samples
        num_samples (int): Total number of samples
                           (initial + burn-in + chain)
        proposal_distribution (scipy.stats.rv_continuous): Proposal distribution
                                                           giving zero-mean deviates
        result_description (dict):  Description of desired results
        samples (numpy.array): Array with all samples
        scale_covariance (float): Scale of covariance matrix
                                  of gaussian proposal distribution
        seed (int): Seed for random number generator
        tune (bool): Tune the scale of covariance
        tune_interval (int): Tune the scale of the covariance every
                             tune_interval-th step

    """

    def __init__(self, global_settings, model, num_burn_in, num_samples,
                 proposal_distribution, result_description, scale_covariance,
                 seed, tune, tune_intervall):
        super().__init__(model, global_settings)
        self.num_burn_in = num_burn_in
        self.num_samples = num_samples
        self.proposal_distribution = proposal_distribution
        self.result_description = result_description

        # check agreement of dimensions of proposal distribution and
        # parameter space
        num_variables = self.model.variables[0].get_number_of_active_variables()
        if num_variables is not self.proposal_distribution.dimension:
            raise ValueError("Dimensions of proposal distribution"
                             "and parameter space do not agree.")

        tot_num_samples = self.num_samples+self.num_burn_in+1
        self.samples = np.zeros((tot_num_samples, num_variables))
        self.tune = tune
        self.tune_interval = tune_intervall
        self.scale_covariance = scale_covariance
        self.seed = seed

        self.accepted = 0
        self.accepted_interval = 0

        self.gamma = 1.

        self.log_likelihood = np.zeros(tot_num_samples)
        self.log_prior = np.zeros(tot_num_samples)
        self.log_posterior = np.zeros(tot_num_samples)

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None,
                                    model=None):
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
            proposal_distribution = mcmc_utils.create_proposal_distribution(proposal_options)
        else:
            raise ValueError(f'Could not find proposal distribution'
                             f' "{name_proposal_distribution}" in input file.')

        tune = method_options.get('tune', False)
        tune_interval = method_options.get('tune_interval', 100)

        return cls(global_settings=global_settings,
                   model=model,
                   num_burn_in=method_options['num_burn_in'],
                   num_samples=method_options['num_samples'],
                   proposal_distribution=proposal_distribution,
                   result_description=result_description,
                   scale_covariance=method_options['scale_covariance'],
                   seed=method_options['seed'],
                   tune=tune,
                   tune_intervall=tune_interval)

    def eval_model(self):
        """ Evaluate model at current sample. """
        result_dict = self.model.evaluate()
        return result_dict

    def eval_log_prior(self, sample):
        """
        Evaluate natural logarithm of prior at sample.

        Note: we assume a multiplicative split of prior pdf
        """

        log_prior = 0.
        i = 0
        for _, variable in self.model.variables[0].variables.items():
            log_prior += variable['distribution'].logpdf(sample[i])
            i += 1

        return log_prior

    def eval_log_likelihood(self, sample):
        """ Evaluate natural logarithm of likelihood at sample. """

        self.model.update_model_from_sample(sample)
        log_likelihood = self.eval_model()['mean']

        return log_likelihood

    def do_mh_step(self, step_id):
        """ Metropolis (Hastings) step. """

        # tune covariance of proposal
        if not step_id % self.tune_interval and self.tune:
            accept_rate_interval = self.accepted_interval / self.tune_interval
            # TODO: decide if main iterator or not (i.e., kernel)
            #print(f"Current acceptance rate: {accept_rate_interval}.")
            self.scale_covariance = mcmc_utils.tune_scale_covariance(self.scale_covariance,
                                                                     accept_rate_interval)
            self.accepted_interval = 0

        cur_sample = self.samples[step_id-1]
        delta_proposal = self.proposal_distribution.draw() * self.scale_covariance
        proposal = cur_sample + delta_proposal

        log_likelihood_prop = self.eval_log_likelihood(proposal)
        log_prior_prop = self.eval_log_prior(proposal)

        log_posterior_prop = log_likelihood_prop * self.gamma + log_prior_prop
        log_accept_prob = log_posterior_prop - self.log_posterior[step_id-1]

        new_sample, accepted = mcmc_utils.mh_select(log_accept_prob, cur_sample, proposal)

        self.accepted += accepted
        self.accepted_interval += accepted

        self.samples[step_id] = new_sample
        if accepted:
            self.log_likelihood[step_id] = log_likelihood_prop
            self.log_prior[step_id] = log_prior_prop
            self.log_posterior[step_id] = log_posterior_prop
        else:
            self.log_likelihood[step_id] = self.log_likelihood[step_id-1]
            self.log_prior[step_id] = self.log_prior[step_id-1]
            self.log_posterior[step_id] = self.log_posterior[step_id-1]

    def initialize_run(self, initial_sample=None, initial_log_like=None, initial_log_prior=None, gamma=1.0):
        """ Draw initial sample. """

        # TODO: decide if main iterator or not
        #print("Initialize Metropolis-Hastings run.")

        # TODO: check conditions (either all are None or none is None)
        if initial_sample is None or initial_log_like is None or initial_log_prior is None:
            np.random.seed(self.seed)

            # draw initial sample from prior distribution
            initial_sample = np.array([variable['distribution'].rvs(size=1)
                                        for model_variable in self.model.variables
                                        for variable_name, variable
                                        in model_variable.variables.items()]).T
            initial_log_like = self.eval_log_likelihood(initial_sample)
            initial_log_prior = self.eval_log_prior(initial_sample)

        self.gamma = gamma

        self.samples[0] = initial_sample
        self.log_likelihood[0] = initial_log_like
        self.log_prior[0] = initial_log_prior

        self.log_posterior[0] = self.log_likelihood[0] * self.gamma + self.log_prior[0]


    def core_run(self):
        """
        Core run of Metropolis-Hastings iterator

        1.) Burn-in phase
        2.) Sampling phase
        """
        # TODO: decide if main iterator or not
        #print('Welcome to Metropolis-Hastings core run.')

        # Burn-in phase
        for i in range(1, self.num_burn_in + 1):
            self.do_mh_step(i)

        if self.num_burn_in:
            burn_in_accept_rate = self.accepted / self.num_burn_in
            print("Acceptance rate during burn in: {0}".format(burn_in_accept_rate))
        # reset number of accepted samples
        self.accepted = 0
        self.accepted_interval = 0

        # Sampling phase
        for i in range(self.num_burn_in + 1, self.num_burn_in + self.num_samples + 1):
            self.do_mh_step(i)

    def post_run(self):
        """ Analyze the resulting chain. """

        if self.result_description:
            as_mcmc_kernel = self.result_description.get('as_mcmc_kernel', False)
            if as_mcmc_kernel:
                # the iterator is used as MCMC kernel for the Sequential Monte Carlo iterator
                return [self.samples[-1],
                        self.log_likelihood[-1],
                        self.log_prior[-1],
                        self.log_posterior[-1]]
            else:
                initial_sample = self.samples[0]
                chain_burn_in = self.samples[1 : self.num_burn_in + 1]
                chain = self.samples[self.num_burn_in + 1:self.num_samples + self.num_burn_in + 1]

                accept_rate = self.accepted / self.num_samples

                # process output takes a dict as input with key 'mean'
                results = process_ouputs({'mean': chain,
                                          'accept_rate': accept_rate,
                                          'chain_burn_in': chain_burn_in,
                                          'initial_sample': initial_sample,
                                          'log_likelihood' : self.log_likelihood,
                                          'log_prior' : self.log_prior,
                                          'log_posterior' : self.log_posterior
                                         },
                                         self.result_description)
                if self.result_description["write_results"]:
                    write_results(results,
                                  self.global_settings["output_dir"],
                                  self.global_settings["experiment_name"])

                print("Acceptance rate: {}".format(accept_rate))
                print(f"Covariance of proposal: {self.scale_covariance * self.proposal_distribution.covariance}")
                print("Size of outputs {}".format(chain.shape))
                print("\tmean±std: {}±{}".format(results.get('mean', None),
                                                 np.sqrt(results.get('var', None))))
                print("\tvar: {}".format(results.get('var', None)))
                print("\tcov: {}".format(results.get('cov', np.array(None)).tolist()))
