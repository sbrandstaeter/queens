import numpy as np
from pqueens.iterators.iterator import Iterator
from pqueens.models.model import Model
from pqueens.utils.mcmc_utils import create_proposal_distribution, mh_select
from pqueens.utils.process_outputs import process_ouputs
from pqueens.utils.process_outputs import write_results

class MetropolisHastingsIterator(Iterator):
    """ Basic Markov Chain Monte Carlo Iterator based on Metropolis-Hastings Algorithm

    Attributes:
        accepted (int): number of accepted proposals
        log_likelihood (np.array): log of pdf of likelihood at samples
        log_posterior (np.array): log of pdf of posterior at samples
        log_prior (np.array): log of pdf of prior at samples
        num_burn_in (int): Number of burnin samples
        num_samples (int): Number of samples in chain (chain length)
        proposal_distribution (scipy.stats.rv_continuous): Proposal distribution, that gives zero-mean deviates
        proposal_covariance (numpy.array): covariance matrix of proposal distribution
        result_description (dict):  Description of desired results
        samples (numpy.array): Array with all samples
        scale_covariance (float): Scale of covariance of gaussian proposal distribution
        seed (int): Seed for random number generator

    """

    def __init__(self, global_settings, model, num_burn_in, num_samples,
                 proposal_distribution, result_description, scale_covariance,
                 seed):
        super().__init__(model, global_settings)
        self.num_burn_in = num_burn_in
        self.num_samples = num_samples
        self.proposal_distribution = proposal_distribution
        self.result_description = result_description

        # check if dimensions of proposal distribution and parameter space agree
        num_variables = self.model.variables[0].get_number_of_active_variables()
        if num_variables is not self.proposal_distribution.dimension:
            raise ValueError("Dimensions of proposal distribution"
                             "and parameter space do not agree.")

        tot_num_samples = self.num_samples+self.num_burn_in+1
        self.samples = np.zeros((tot_num_samples, num_variables))
        self.scale_covariance = scale_covariance
        self.seed = seed

        self.accepted = 0

        self.log_likelihood = np.zeros(tot_num_samples)
        self.log_prior = np.zeros(tot_num_samples)
        self.log_posterior = np.zeros(tot_num_samples)

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None,
                                    model=None):
        """ Create Metropolis-Hastings iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
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
            proposal_distribution = create_proposal_distribution(proposal_options)
        else:
            raise ValueError('Could not find proposal distribution "{0}" in input file.'.format(name_proposal_distribution))

        return cls(global_settings=global_settings,
                   model=model,
                   num_burn_in=method_options['num_burn_in'],
                   num_samples=method_options['num_samples'],
                   proposal_distribution=proposal_distribution,
                   result_description=result_description,
                   scale_covariance=method_options['scale_covariance'],
                   seed=method_options['seed'])

    def eval_model(self):
        """ Evaluate the model """
        result_dict = self.model.evaluate()
        return result_dict

    def eval_log_prior(self, sample):
        """ Evaluate the log of the prior at sample
        Note: we assume a multiplicative split of the prior
        """
        log_prior = 0.
        i = 0
        for _, variable in self.model.variables[0].variables.items():
            log_prior += variable['distribution'].logpdf(sample[i])
            i += 1

        return log_prior

    def eval_log_likelihood(self, sample):
        """ Evaluate the log of the likelihood at sample"""

        self.model.update_model_from_sample(sample)
        log_likelihood = self.eval_model()['mean']

        return log_likelihood

    def do_mh_step(self, step_id):
        """ Metropolis (Hastings) Step"""

        cur_sample = self.samples[step_id-1]
        delta_proposal = self.proposal_distribution.draw() * self.scale_covariance
        proposal = cur_sample + delta_proposal

        log_likelihood_prop = self.eval_log_likelihood(proposal)
        log_prior_prop = self.eval_log_prior(proposal)

        log_posterior_prop = log_likelihood_prop + log_prior_prop
        log_accept_prob = log_posterior_prop - self.log_posterior[step_id-1]

        new_sample, accepted = mh_select(log_accept_prob, cur_sample, proposal)

        self.accepted += accepted

        self.samples[step_id] = new_sample
        if accepted:
            self.log_likelihood[step_id] = log_likelihood_prop
            self.log_prior[step_id] = log_prior_prop
            self.log_posterior[step_id] = log_posterior_prop
        else:
            self.log_likelihood[step_id] = self.log_likelihood[step_id-1]
            self.log_prior[step_id] = self.log_prior[step_id-1]
            self.log_posterior[step_id] = self.log_posterior[step_id-1]

    def pre_run(self):
        """ Draw initial sample """

        print("Welcome to Metropolis-Hastings Pre Run")
        np.random.seed(self.seed)

        # draw initial sample from prior distribution
        self.samples[0] = np.array([variable['distribution'].rvs(size=1)
                                    for model_variable in self.model.variables
                                    for variable_name, variable
                                    in model_variable.variables.items()]).T

        self.log_likelihood[0] = self.eval_log_likelihood(self.samples[0])
        self.log_prior[0] = self.eval_log_prior(self.samples[0])
        self.log_posterior[0] = self.log_likelihood[0] + self.log_prior[0]

    def core_run(self):
        print('Welcome to Metropolis-Hastings Core Run.')

        for i in range(1, self.num_burn_in + 1):
            self.do_mh_step(i)

        if self.num_burn_in is not 0:
            burnin_accept_rate = self.accepted / self.num_burn_in
            print("Acceptance rate during burn in: {0}".format(burnin_accept_rate))
        # reset number of accepted samples
        self.accepted = 0

        for i in range(self.num_burn_in + 1, self.num_burn_in + self.num_samples + 1):
            self.do_mh_step(i)

    def post_run(self):
        """ Analyze the results """
        chain = self.samples[self.num_burn_in + 1:self.num_samples + self.num_burn_in + 1]

        accept_rate = self.accepted / self.num_samples

        # process output requires take a dict as input with label 'mean'
        if self.result_description is not None:
            results = process_ouputs({'mean': chain}, self.result_description)
            if self.result_description["write_results"] is True:
                write_results(results,
                              self.global_settings["output_dir"],
                              self.global_settings["experiment_name"])

        print("Acceptance rate: {}".format(accept_rate))
        print("Size of outputs {}".format(chain.shape))
        print("\tmean±std: {}±{}".format(results['mean'], np.sqrt(results['var'])))
        print("\tvar: {}".format(results['var']))

