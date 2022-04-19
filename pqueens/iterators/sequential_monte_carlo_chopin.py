"""Sequential Monte Carlo implementation using 'particles' package."""

import logging

import numpy as np
import particles
from particles import collectors as col
from particles import distributions as dists
from particles.smc_samplers import AdaptiveTempering, TemperingBridge

from pqueens.iterators.iterator import Iterator
from pqueens.models import from_config_create_model
from pqueens.utils import smc_utils
from pqueens.utils.process_outputs import process_ouputs, write_results

_logger = logging.getLogger(__name__)


class SequentialMonteCarloChopinIterator(Iterator):
    """Sequential Monte Carlo algorithm from Chopin et al.

    Sequential Monte Carlo algorithm based on the book [1] (especially chapter 17)
    and the particles library (https://github.com/nchopin/particles/)


    References:
        [1]: Chopin N. and Papaspiliopoulos O. (2020), An Introduction to Sequential Monte Carlo,
             10.1007/978-3-030-47845-2 , Springer.

    Attributes:
        global_settings (dict): Global settings of the QUEENS simulations
        model (obj): Underlying simulation model on which the inverse analysis is conducted
        result_description (dict): Settings for storing and visualizing the results
        num_particles (int): Number of particles
        num_variables (int): Number of primary variables
        result_description (dict):  Description of desired results
        seed (int): Seed for random number generator
        max_feval (int): Maximum number of model calls
        n_sims (int): Number of model calls
        prior (object): Particles Prior object
        smc_obj (object): Particles SMC object
        random_variable_keys (list): Random variables names
        resampling_threshold (float): Ratio of ESS to partice number at which to resample
        resampling_method (str): Resampling method implemented in particles
        feynman_kac_model (str): Feynman Kac model for the smc object
        num_rejuvenation_steps (int): Number of rejuvenation steps (e.g. MCMC steps)
        waste_free (bool): if True, all intermediate Markov steps are kept
    """

    def __init__(
        self,
        global_settings,
        model,
        num_particles,
        result_description,
        seed,
        n_sims,
        max_feval,
        prior,
        smc_obj,
        random_variable_keys,
        resampling_threshold,
        resampling_method,
        feynman_kac_model,
        num_rejuvenation_steps,
        waste_free,
    ):
        """Initialize the SMC iterator.

        Args:
            global_settings (dict): Global settings of the QUEENS simulations
            model (obj): Underlying simulation model on which the inverse analysis is conducted
            num_particles (int): Number of particles
            result_description (dict): Settings for storing and visualizing the results
            seed (int): Seed for random number generator
            n_sims (int): Number of model calls
            max_feval (int): Maximum number of model calls
            prior (object): Particles Prior object
            smc_obj (object): Particles SMC object
            random_variable_keys (list): Random variables names
            resampling_threshold (float): Ratio of ESS to partice number at which to resample
            resampling_method (str): Resampling method implemented in particles
            feynman_kac_model (str): Feynman Kac model for the smc object
            num_rejuvenation_steps (int): Number of rejuvenation steps (e.g. MCMC steps)
            waste_free (bool): if True, all intermediate Markov steps are kept
        """
        super().__init__(model, global_settings)
        self.result_description = result_description
        self.seed = seed
        self.num_particles = num_particles
        self.num_variables = self.model.variables[0].get_number_of_active_variables()
        self.n_sims = n_sims
        self.max_feval = max_feval
        self.prior = prior
        self.smc_obj = smc_obj
        self.random_variable_keys = random_variable_keys
        self.resampling_threshold = resampling_threshold
        self.resampling_method = resampling_method
        self.feynman_kac_model = feynman_kac_model
        self.num_rejuvenation_steps = num_rejuvenation_steps
        self.waste_free = waste_free

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create SMC Chopin iterator from problem description.

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator
            model (model):       Model to use

        Returns:
            iterator: SequentialMonteCarloChopinIterator object
        """
        method_options = config[iterator_name]['method_options']
        if model is None:
            model_name = method_options['model']
            model = from_config_create_model(model_name, config)

        result_description = method_options.get('result_description', None)
        global_settings = config.get('global_settings', None)

        num_particles = method_options.get("num_particles")
        max_feval = method_options.get("max_feval")
        smc_seed = method_options.get("seed")
        resampling_threshold = method_options.get("resampling_threshold")
        resampling_method = method_options.get("resampling_method")
        feynman_kac_model = method_options.get("feynman_kac_model")
        num_rejuvenation_steps = method_options.get("num_rejuvenation_steps")
        print(num_rejuvenation_steps)
        waste_free = method_options.get("waste_free")
        return cls(
            global_settings=global_settings,
            model=model,
            num_particles=num_particles,
            result_description=result_description,
            seed=smc_seed,
            max_feval=max_feval,
            resampling_threshold=resampling_threshold,
            resampling_method=resampling_method,
            feynman_kac_model=feynman_kac_model,
            num_rejuvenation_steps=num_rejuvenation_steps,
            waste_free=waste_free,
            n_sims=0,
            prior=None,
            smc_obj=None,
            random_variable_keys=[],
        )

    def eval_model(self):
        """Evaluate model at current sample."""
        result_dict = self.model.evaluate()
        return result_dict

    def eval_log_likelihood(self, samples):
        """Evaluate natural logarithm of likelihood at sample.

        Args:
            samples (np.array): Samples/particles of the SMC.
        """
        self.model.update_model_from_sample_batch(samples)
        log_likelihood = self.eval_model()
        return log_likelihood

    def _initialize_prior_model(self):
        """Initialize the prior model form the problem description.

        Returns:
            None
        """
        # Read design of prior
        input_dict = self.model.get_parameter()
        # get random variables
        random_variables = input_dict.get('random_variables', None)
        # get random fields
        random_fields = input_dict.get('random_fields', None)

        if random_fields:
            raise NotImplementedError(
                'Particles SMC for random fields is not yet implemented! Abort...'
            )
        # Generate prior using the particles library
        prior_dict = {}
        for rv in random_variables.keys():
            rv_options = random_variables[rv]
            distribution = rv_options.get("distribution")
            distribution_params = rv_options.get("distribution_parameter")
            if distribution == "normal":
                loc = distribution_params[0]
                scale = distribution_params[1] ** 0.5
                prior_dict.update({rv: dists.Normal(loc=loc, scale=scale)})
            elif distribution == "uniform":
                a = distribution_params[0]
                b = distribution_params[1]
                prior_dict.update({rv: dists.Uniform(a=a, b=b)})
            else:
                raise NotImplementedError(
                    f"Currently the priors are only allowed to be normal or uniform"
                )
            self.random_variable_keys.append(rv)
        self.prior = dists.StructDist(prior_dict)

    def initialize_feynman_kac(self, static_model):
        """Initialize the Feynman Kac model for the SMC approach.

        Args:
            static_model (StaticModel): Static model from the particles library

        Returns:
            feynman_kac_model (FKSMCsampler): Model for the smc object
        """
        if self.feynman_kac_model == "adaptive_tempering":
            feynman_kac_model = AdaptiveTempering(
                static_model,
                ESSrmin=self.resampling_threshold,
                wastefree=self.waste_free,
                len_chain=self.num_rejuvenation_steps + 1,  # is handle this way in the library
            )
        else:
            raise NotImplementedError(
                f"The allowed Feynman Kac models are: 'tempering' and 'adaptive_tempering'"
            )
        return feynman_kac_model

    def initialize_run(self):
        """Draw initial sample."""
        _logger.info("Initialize run.")
        self._initialize_prior_model()
        np.random.seed(self.seed)

        # Likelihood function for the static model based on the QUEENS function
        log_likelihood = lambda x: self.eval_log_likelihood(x)

        # Static model for the Feynman Kac model
        static_model = smc_utils.StaticStateSpaceModel(
            data=None,
            prior=self.prior,
            likelihood_model=log_likelihood,
            random_variable_keys=self.random_variable_keys,
        )

        # Feynman Kac model for the SMC algorithm
        feynman_kac_model = self.initialize_feynman_kac(static_model)

        # SMC object
        self.smc_obj = particles.SMC(
            fk=feynman_kac_model,
            N=self.num_particles,
            verbose=True,
            collect=[col.Moments()],
            resampling=self.resampling_method,
            qmc=False,  # QMC can not be used in this static setting in particles (currently)
        )

    def core_run(self):
        """Core run of Sequential Monte Carlo iterator.

        The particles library is generator based. Hence one step of the
        SMC algorithm is done using next(self.smc). As the next()
        function is called during the for loop, we only need to add some
        logging and check if the number of model runs is exceeded.
        """
        _logger.info('Welcome to SMC (particles) core run.')

        for _ in self.smc_obj:
            _logger.info(f"SMC step {self.smc_obj.t-1}")
            self.n_sims = self.smc_obj.fk.model.n_sims
            _logger.info(f"Number of forward runs {self.n_sims}")
            _logger.info("-" * 70)
            if self.n_sims >= self.max_feval:
                _logger.warning(f"Maximum number of model evaluations reached!")
                _logger.warning(f"Stopping SMC...")
                break

    def post_run(self):
        """Analyze the resulting importance sample."""
        # SMC data
        particles = self.smc_obj.fk.model.particles_array_to_numpy(self.smc_obj.X.theta)
        weights = self.smc_obj.W.reshape(-1, 1)

        # First and second moment
        mean = self.smc_obj.fk.model.particles_array_to_numpy(
            self.smc_obj.summaries.moments[-1]["mean"]
        )[0]
        variance = self.smc_obj.fk.model.particles_array_to_numpy(
            self.smc_obj.summaries.moments[-1]["var"]
        )[0]
        if self.result_description:
            results = process_ouputs(
                {
                    'particles': particles,
                    'weights': weights,
                    'log_posterior': self.smc_obj.X.lpost,
                    "mean": mean,
                    "var": variance,
                    "n_sims": self.n_sims,
                },
                self.result_description,
            )
            if self.result_description["write_results"]:
                write_results(
                    results,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )
            _logger.info("Post run data exported!")
