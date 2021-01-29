"""
Sequential Monte Carlo algorithm


References:
[1]: Del Moral, P., Doucet, A. and Jasra, A. (2007)
    ‘Sequential monte carlo for bayesian computation’,
     in Bernardo, J. M. et al. (eds) Bayesian Statistics 8.
     Oxford University Press, pp. 1–34.
[2]: Koutsourelakis, P. S. (2009)
    ‘A multi-resolution, non-parametric, Bayesian framework for
     identification of spatially-varying model parameters’,
     Journal of Computational Physics, 228(17), pp. 6184–6211.
     doi: 10.1016/j.jcp.2009.05.016.
[3]: Minson, S. E., Simons, M. and Beck, J. L. (2013)
    ‘Bayesian inversion for finite fault earthquake source models
     I-theory and algorithm’,
     Geophysical Journal International, 194(3), pp. 1701–1726.
     doi: 10.1093/gji/ggt180.
[4]: Del Moral, P., Doucet, A. and Jasra, A. (2006)
     ‘Sequential Monte Carlo samplers’,
     Journal of the Royal Statistical Society.
     Series B: Statistical Methodology.
     Blackwell Publishing Ltd, 68(3), pp. 411–436.
     doi: 10.1111/j.1467-9868.2006.00553.x.
"""
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy

from pqueens.iterators.iterator import Iterator
from pqueens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from pqueens.models.model import Model
from pqueens.utils import smc_utils
from pqueens.utils.process_outputs import process_ouputs
from pqueens.utils.process_outputs import write_results


class SequentialMonteCarloIterator(Iterator):
    """
    Iterator based on Sequential Monte Carlo algorithm

    The Sequential Monte Carlo algorithm is a very general algorithm for
    sampling from complex, intractable probability distributions from
    which direct sampling is difficult or impossible.
    The implemented version is based on [1, 2, 3, 4].

    Attributes:
        a (float): parameter for the scaling of the covariance matrix of the proposal distribution
                   of the MCMC kernel
        accepted (int): number of accepted proposals
        b (float): parameter for the scaling of the covariance matrix of the proposal distribution
                   of the MCMC kernel
        ess (list): list storing the values of the effective sample size
        ess_cur (float): current effective sample size
        gamma_cur (float): current tempering parameter (sometimes called (reciprocal) temperature)
        gammas (list): list to store values of the tempering parameter
        log_likelihood (ndarray): log of pdf of likelihood at particles
        log_posterior (ndarray): log of pdf of posterior at particles
        log_prior (ndarray): log of pdf of prior at particles
        num_particles (int): Number of particles
        num_variables (int): Number of primary variables
        mcmc_kernel (MetropolisHastingsIterator): forward kernel for the rejuvenation steps
        particles (ndarray): array with holding the current particles
        proposal_distribution (): Proposal distribution giving zero-mean deviates
        plot_trace_every (int): print the current tace every plot_trace_every-th iteration
                                default: 0 (do not print the trace)
        result_description (dict):  Description of desired results
        seed (int): Seed for random number generator
        temper (function): tempering function that defines the transition to the goal distribution
        weights (ndarray): array holding the current weights
    """

    def __init__(
        self,
        global_settings,
        mcmc_kernel,
        model,
        num_particles,
        plot_trace_every,
        result_description,
        seed,
        temper_type,
    ):
        super().__init__(model, global_settings)
        self.plot_trace_every = plot_trace_every
        self.result_description = result_description
        self.seed = seed

        self.mcmc_kernel = mcmc_kernel

        self.num_particles = num_particles

        # check agreement of dimensions of proposal distribution and
        # parameter space
        self.num_variables = self.model.variables[0].get_number_of_active_variables()

        if self.num_variables is not self.mcmc_kernel.proposal_distribution.dimension:
            raise ValueError(
                "Dimensions of MCMC kernel proposal"
                " distribution and parameter space"
                " do not agree."
            )

        # init particles (location, weights, posterior)
        self.particles = np.zeros((self.num_particles, self.num_variables))
        # TODO: use normalised weights?
        self.weights = np.ones((self.num_particles, 1))

        self.log_likelihood = np.zeros((self.num_particles, 1))
        self.log_prior = np.zeros((self.num_particles, 1))
        self.log_posterior = np.zeros((self.num_particles, 1))

        self.ess = list()
        self.ess_cur = 0.0

        self.temper = smc_utils.temper_factory(temper_type)

        # tempering parameter (linked to counter/ time index)
        self.gamma_cur = 0.0
        self.gammas = list()

        # parameters for the scaling of the covariance matrix
        # values of a an b are taken from [3] p.1706
        self.a = 1.0 / 9.0
        self.b = 8.0 / 9.0

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """
        Create Sequential Monte Carlo iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: SequentialMonteCarloIterator object

        """

        print(
            "Sequential Monte Carlo Iterator for experiment: {0}".format(
                config.get('global_settings').get('experiment_name')
            )
        )
        if iterator_name is None:
            method_options = config['method']['method_options']
        else:
            method_options = config[iterator_name]['method_options']
        if model is None:
            model_name = method_options['model']
            model = Model.from_config_create_model(model_name, config)

        plot_trace_every = method_options.get('plot_trace_every', 0)
        result_description = method_options.get('result_description', None)
        global_settings = config.get('global_settings', None)

        # check sanity of MCMC kernel config
        kernel_options = config.get('MCMC_Kernel', None)
        if kernel_options is None:
            raise ValueError("You need to specify a MCMC Kernel.")
        if ('as_mcmc_kernel' not in kernel_options['method_options']) or (
            not kernel_options['method_options']['as_mcmc_kernel'] is True
        ):
            raise ValueError("MH iterator needs to be specified as MCMC Kernel.")
        if not (
            kernel_options['method_options'].get('num_chains', 1) == method_options['num_particles']
        ):
            warnings.warn(
                "Number of chains in the kernel has to be equal to number of particles:"
                " setting num_chains to num_particles."
            )
            config['MCMC_Kernel']['method_options']['num_chains'] = method_options['num_particles']

        # ensure that the seeds are identical
        kernel_seed = kernel_options['method_options'].get('seed', None)
        smc_seed = method_options['seed']
        if kernel_seed != smc_seed:
            kernel_options['method_options']['seed'] = smc_seed

        temper_type = method_options['temper_type']

        mcmc_kernel = MetropolisHastingsIterator.from_config_create_iterator(
            config, iterator_name='MCMC_Kernel', model=model, temper_type=temper_type
        )

        return cls(
            global_settings=global_settings,
            mcmc_kernel=mcmc_kernel,
            model=model,
            num_particles=method_options['num_particles'],
            plot_trace_every=plot_trace_every,
            result_description=result_description,
            seed=smc_seed,
            temper_type=temper_type,
        )

    def eval_model(self):
        """ Evaluate model at current sample. """
        result_dict = self.model.evaluate()
        return result_dict

    def eval_log_prior(self, sample):
        """
        Evaluate natural logarithm of prior at sample.

        Note: we assume a multiplicative split of prior pdf
        """

        log_prior = 0.0
        i = 0
        for _, variable in self.model.variables[0].variables.items():
            log_prior += variable['distribution'].logpdf(sample[i])
            i += 1

        return log_prior

    def eval_log_likelihood(self, sample):
        """ Evaluate natural logarithm of likelihood at sample. """

        self.model.update_model_from_sample(sample)
        log_likelihood = self.eval_model()

        return log_likelihood

    def initialize_run(self):
        """ Draw initial sample. """

        print("Initialize run.")
        np.random.seed(self.seed)

        # draw initial particles from prior distribution
        for i in range(self.num_particles):
            self.particles[i] = np.array(
                [
                    variable['distribution'].draw(num_draws=1)
                    for model_variable in self.model.variables
                    for variable_name, variable in model_variable.variables.items()
                ]
            ).T
            self.log_likelihood[i] = self.eval_log_likelihood(self.particles[i])
            self.log_prior[i] = self.eval_log_prior(self.particles[i])
            self.log_posterior[i] = self.log_likelihood[i] + self.log_prior[i]

        # initialize importance weights
        self.weights = np.ones((self.num_particles, 1))
        self.ess_cur = self.num_particles
        self.ess.append(self.ess_cur)

        self.gamma_cur = 0.0
        self.gammas.append(self.gamma_cur)

        if self.plot_trace_every:
            self.draw_trace(0)

    def calc_new_weights(self, gamma_new, gamma_old):
        """
        Calculate the weights at new gamma value.

        This is a core equation of the SMC algorithm. See for example
        - Eq.(22) with Eq.(14) in [1]
        - Table 1: (2) in [2]
        - Eq.(31) with Eq.(11) in [4]
        """

        weights_new = self.weights * np.exp(
            self.temper(self.log_prior, self.log_likelihood, gamma_new)
            - self.temper(self.log_prior, self.log_likelihood, gamma_old)
        )
        return weights_new

    def calc_new_ess(self, gamma_new, gamma_old):
        """ Calculate predicted Effective Sample Size at gamma_new. """

        weights_new = self.calc_new_weights(gamma_new, gamma_old)
        ess = smc_utils.calc_ess(weights_new)
        return ess

    def calc_new_gamma(self, gamma_cur):
        """
        Calculate the new gamma value.

        Based on the current gamma, calculate the new gamma such that
        the ESS at the new gamma is equal to zeta times current gamma.
        This ensures only a small reduction of the ESS.
        """
        zeta = 0.95

        def f(gamma_new):
            ess_new = self.calc_new_ess(gamma_new, gamma_cur)

            f = ess_new - zeta * self.ess_cur
            return f

        # TODO: adjust accuracy - method, xtol, rtol, maxiter
        # TODO: new ESS is already calculated in this step
        search_interval = [gamma_cur, 1.0]
        try:
            root_result = scipy.optimize.root_scalar(f, bracket=search_interval, method='toms748')
            gamma_new = root_result.root
        except:
            print(f"Could not find suitable gamma within {search_interval}: setting gamma=1.0")
            gamma_new = 1.0

        return gamma_new

    def update_ess(self, resampled=False):
        """
        Update effective sample size (ess) and store current value

        Based on the current weights, calculate the corresponding ESS.
        Store the new ESS value.
        In case of resampling, the weights have been reset in the
        current time step and therefore also the ESS has to be reset.

        :param resampled: (bool) indicated whether current weights
                                 are base on a resampling step
        :return: None
        """
        self.ess_cur = smc_utils.calc_ess(self.weights)

        if resampled:
            self.ess[-1] = self.ess_cur
        else:
            self.ess.append(self.ess_cur)

    def update_gamma(self, gamma_new):
        """ Update the current gamma value and store old value. """

        self.gamma_cur = gamma_new
        self.gammas.append(self.gamma_cur)

    def update_weights(self, weights_new):
        """ Update the weights to their new values. """

        self.weights = weights_new

    def resample(self):
        """
        Resample particle distribution based on their weights.

        Resampling reduces the variance of the particle approximation by
        eliminating particles with small weights and duplicating
        particles with large weights (see 2.2.1 in [2]).
        """
        normalized_weights = self.weights / np.sum(self.weights)

        # draw from multinomial distribution to decide
        # the frequency of individual particles
        particle_freq = np.random.multinomial(self.num_particles, np.squeeze(normalized_weights))

        idx_list = list()
        for idx, freq in enumerate(particle_freq):
            idx_list += [idx] * freq

        resampled_weights = np.ones((self.num_particles, 1))

        return (
            self.particles[idx_list],
            resampled_weights,
            self.log_likelihood[idx_list],
            self.log_prior[idx_list],
        )

    def core_run(self):
        """ Core run of Sequential Monte Carlo iterator. """

        print('Welcome to SMC core run.')

        # counter
        step = 0
        # average accept rate of MCMC kernel
        avg_accept = 1.0
        while self.gamma_cur < 1:
            step += 1

            # Adapt step size
            self.update_gamma(self.calc_new_gamma(self.gamma_cur))

            # Reweigh
            self.update_weights(self.calc_new_weights(self.gamma_cur, self.gammas[-2]))
            self.update_ess()

            # Resample
            if self.ess_cur <= 0.5 * self.num_particles:
                print("Resampling...")
                (
                    particles_resampled,
                    weights_resampled,
                    log_like_resampled,
                    log_prior_resampled,
                ) = self.resample()

                # update algorithm parameters
                self.particles = particles_resampled
                self.log_likelihood = log_like_resampled
                self.log_prior = log_prior_resampled
                self.log_posterior = self.log_likelihood + self.log_prior
                self.update_weights(weights_resampled)

                self.update_ess(resampled=True)

            print(f"step {step} gamma: {self.gamma_cur:.5} ESS: {self.ess_cur:.5}")

            # estimate current covariance matrix
            cov_mat = np.atleast_2d(
                np.cov(self.particles, ddof=0, aweights=np.squeeze(self.weights), rowvar=False)
            )

            # scale covariance based on average acceptance rate of last rejuvenation step
            scale_prop_cov = self.a + self.b * avg_accept
            cov_mat *= scale_prop_cov ** 2

            # Rejuvenate
            self.mcmc_kernel.initialize_run(
                self.particles, self.log_likelihood, self.log_prior, self.gamma_cur, cov_mat
            )
            self.mcmc_kernel.core_run()
            (
                self.particles,
                self.log_likelihood,
                self.log_prior,
                self.log_posterior,
                avg_accept,
            ) = self.mcmc_kernel.post_run()

            # plot the trace every plot_trace_every-th iteration
            if self.plot_trace_every and not step % self.plot_trace_every:
                self.draw_trace(step)

    def post_run(self):
        """ Analyze the resulting importance sample. """

        normalized_weights = self.weights / np.sum(self.weights)

        particles_resampled, _, _, _ = self.resample()
        if self.result_description:
            # TODO
            # interpret the resampled particles as a single markov chain -> in accordance with the
            # Metropolis Hastings iterator add a dimension to the numpy array
            # this enables the calculation of the covariance matrix
            results = process_ouputs(
                {
                    'mean': particles_resampled[:, np.newaxis, :],
                    'particles': self.particles,
                    'weights': normalized_weights,
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

            if self.result_description["plot_results"]:
                self.draw_trace('final')

            mean = np.sum(np.multiply(np.squeeze(normalized_weights), self.particles.T).T, axis=0)
            var = np.sum(
                np.multiply(np.squeeze(normalized_weights), np.power(self.particles - mean, 2).T).T,
                axis=0,
            )
            std = np.sqrt(var)

            print("\tESS: {}".format(self.ess_cur))
            print(f"\tIS mean±std: {mean}±{std}")

            print(
                "\tmean±std: {}±{}".format(
                    results.get('mean', np.nan), np.sqrt(results.get('var', np.nan))
                )
            )
            print("\tvar: {}".format(results.get('var', np.nan)))
            print("\tcov: {}".format(results.get('cov', np.nan)))

    def draw_trace(self, step):
        """
        Plot the trace of the current particle approximation.

        :param step: (int) current step index
        :return: None
        """

        particles_resampled, _, _, _ = self.resample()
        data_dict = {
            variable_name: particles_resampled[:, i]
            for model_variable in self.model.variables
            for i, (variable_name, variable) in enumerate(model_variable.variables.items())
        }
        inference_data = az.convert_to_inference_data(data_dict)
        az.plot_trace(inference_data)
        plt.savefig(
            f"{self.global_settings['output_dir']}/{self.global_settings['experiment_name']}"
            + f"_trace_{step}.png"
        )
        plt.close("all")
