"""Black box variational inference iterator."""
import logging
import time

import numpy as np

import pqueens.database.database as DB_module
import pqueens.visualization.variational_inference_visualization as vis
from pqueens.iterators.iterator import Iterator
from pqueens.models import from_config_create_model
from pqueens.utils import mcmc_utils, variational_inference_utils
from pqueens.utils.process_outputs import write_results
from pqueens.utils.stochastic_optimizer import StochasticOptimizer

_logger = logging.getLogger(__name__)


class BBVIIterator(Iterator):
    """Black box variational inference (BBVI) iterator.

    For Bayesian inverse problems. BBVI does not require model gradients and can hence be used with
    any simulation model and without the need for adjoints implementations. The algorithm is based
    on [1]. The expectations for the gradient computations are computed using an importance
    sampling approach where the IS-distribution is constructed as a mixture of the variational
    distribution from previous iterations (similar as in [2]).

    Keep in mind:
        This algorithm requires the logpdf of the variational distribution to be differentiable w.r.
        t. the variational parameters. This is not the case for certain distributions, e.g. uniform
        distribution, and can therefore not be used in combination with this algorithm (see [3]
        page 13)!

    References:
        [1]: Ranganath, Rajesh, Sean Gerrish, and David M. Blei. "Black Box Variational Inference."
             Proceedings of the Seventeenth International Conference on Artificial Intelligence
             and Statistics. 2014.
        [2]: Arenz, Neumann & Zhong. "Efficient Gradient-Free Variational Inference using Policy
             Search." Proceedings of the 35th International Conference on Machine Learning (2018) in
             PMLR 80:234-243
        [3]: Mohamed et al. "Monte Carlo Gradient Estimation in Machine Learning". Journal of
             Machine Learning Research. 21(132):1−62, 2020.

    Attributes:
        global_settings (dict): Global settings of the QUEENS simulations
        model (obj): Underlying simulation model on which the inverse analysis is conducted
        result_description (dict): Settings for storing and visualizing the results
        db (obj): QUEENS database object
        experiment_name (str): Name of the QUEENS simulation
        variational_family (str): Density type for variatonal approximation
        variational_params_initialization_approach (str): Flag to decide how to initialize the
                                                          variational paramaters
        n_samples_per_iter (int): Batch size per iteration (number of simulations per iteration to
                                  estimate the involved expectations)
        variational_transformation (str): String encoding the transformation that will be applied to
                                          the variational density
        random_seed (int): Seed for the random number generators
        max_feval (int): Maximum number of simulation runs for this analysis
        num_variables (int): Actual number of model input variables that should be calibrated
        memory (int): Number of previous iterations that should be included in the MC ELBO
                      gradient estimations. For memory=0 the algorithm reduces to standard the
                      standard BBVI algorithm. (Better variable name is welcome)
        natural_gradient_bool (boolean): True if natural gradient should be used
        fim_dampening_bool (boolean): True if FIM dampening should be used
        fim_decay_start_iter (float): Iteration at which the FIM dampening is started
        fim_dampening_coefficient (float): Initial nugget term value for the FIM dampening
        fim_dampening_lower_bound (float): Lower bound on the FIM dampening coefficient
        export_quantities_over_iter (boolean): True if data (variational_params, elbo, ESS) should
                                               be exported in the pickle file
        control_variates_scaling_type (str): Flag to decide how to compute control variate scaling
        loo_cv_bool (boolean): True if leave-one-out procedure is used for the control variate
                               scaling estimations. Is quite slow!
        n_sims (int): Number of probabilistic model calls
        variational_distribution_obj (VariationalDistribution): Variational distribution object
        variational_params (np.array): Rowvector containing the variatonal parameters
        f_mat (np.array): Column-wise ELBO gradient samples
        h_mat (np.array): Column-wise control variate
        elbo_list (list): ELBO value of every iteration
        log_variational_mat (np.array): Logpdf evaluations of the variational distribution
        grad_params_log_variational_mat (np.array): Column-wise grad params logpdf (score function)
                                                    of the variational distribution
        log_posterior_unnormalized (np.array): Rowvector logarithmic probabilistic model evaluation
                                               (generally unnormalized)
        prior_obj_list (list): List containing objects of prior models for the random input. The
                               list is ordered in the same way as the random input definition in
                               the input file
        samples_list (list): List of samples from previous iterations for the ISMC gradient
        parameter_list (list): List of parameters from previous iterations for the ISMC gradient
        log_posterior_unnormalized_list (list): List of probabilistic model evaluations from
                                                previous iterations for the ISMC gradient
        ess_list (list): List containing the effective sample size for every iteration (in case IS
                         is used)
        noise_list (list): Gaussian likelihood noise variance values.
        variational_params_list (list): List of parameters from first to last iteration
        model_eval_iteration_period (int): If the iteration number is a multiple of this number
                                           the probabilistic model is sampled independent of the
                                           other conditions
        stochastic_optimizer (obj): QUEENS stochastic optimizer object
    """

    def __init__(
        self,
        global_settings,
        model,
        result_description,
        db,
        experiment_name,
        variational_family,
        variational_params_initialization_approach,
        n_samples_per_iter,
        variational_transformation,
        random_seed,
        max_feval,
        num_variables,
        memory,
        natural_gradient_bool,
        fim_dampening_bool,
        fim_decay_start_iter,
        fim_dampening_coefficient,
        fim_dampening_lower_bound,
        export_quantities_over_iter,
        control_variates_scaling_type,
        loo_cv_bool,
        variational_distribution_obj,
        stochastic_optimizer,
        model_eval_iteration_period,
        resample,
    ):
        """Initialize BBVI iterator.

        Args:
            global_settings (dict): Global settings of the QUEENS simulations
            model (obj): Underlying simulation model on which the inverse analysis is conducted
            result_description (dict): Settings for storing and visualizing the results
            db (obj): QUEENS database object
            experiment_name (str): Name of the QUEENS simulation
            variational_family (str): Density type for variatonal approximation
            variational_params_initialization_approach (str): Flag to decide how to initialize the
                                                              variational paramaters
            n_samples_per_iter (int): Batch size per iteration (number of simulations per iteration
                                                to estimate the involved expectations)
            variational_transformation (str): String encoding the transformation that will be
                                              applied to
                                              the variational density
            random_seed (int): Seed for the random number generators
            max_feval (int): Maximum number of simulation runs for this analysis
            num_variables (int): Actual number of model input variables that should be calibrated
            memory (int): Number of previous iterations that should be included in the MC ELBO
                          gradient estimations. For memory=0 the algorithm reduces to standard the
                          standard BBVI algorithm. (Better variable name is welcome)
            natural_gradient_bool (boolean): True if natural gradient should be used
            fim_dampening_bool (boolean): True if FIM dampening should be used
            fim_decay_start_iter (float): Iteration at which the FIM dampening is started
            fim_dampening_coefficient (float): Initial nugget term value for the FIM dampening
            fim_dampening_lower_bound (float): Lower bound on the FIM dampening coefficient
            export_quantities_over_iter (boolean): True if data (variational_params, elbo, ESS)
                                                   should be exported in the pickle file
            control_variates_scaling_type (str): Flag to decide how to compute control variate
                                                scaling
            loo_cv_bool (boolean): True if leave-one-out procedure is used for the control variate
                                   scaling estimations. Is quite slow!
            variational_distribution_obj (VariationalDistribution): Variational distribution object
            variational_params_list (list): List of parameters from first to last iteration
            model_eval_iteration_period (int): If the iteration number is a multiple of this number
                                               the probabilistic model is sampled independent of the
                                               other conditions
            resample (bool): True is resampling should be used
            stochastic_optimizer (obj): QUEENS stochastic optimizer object
        Returns:
            bbvi_obj (obj): Instance of the BBVIIterator
        """
        super().__init__(model, global_settings)

        self.result_description = result_description
        self.db = db
        self.experiment_name = experiment_name
        self.variational_params_initialization_approach = variational_params_initialization_approach
        self.n_samples_per_iter = n_samples_per_iter
        self.variational_transformation = variational_transformation
        self.natural_gradient_bool = natural_gradient_bool
        self.fim_decay_start_iter = fim_decay_start_iter
        self.fim_dampening_coefficient = fim_dampening_coefficient
        self.fim_dampening_lower_bound = fim_dampening_lower_bound
        self.fim_dampening_bool = fim_dampening_bool
        self.export_quantities_over_iter = export_quantities_over_iter
        self.control_variates_scaling_type = control_variates_scaling_type
        self.loo_cv_bool = loo_cv_bool
        self.random_seed = random_seed
        self.max_feval = max_feval
        self.num_variables = num_variables
        self.memory = memory
        self.variational_family = variational_family
        self.stochastic_optimizer = stochastic_optimizer
        self.model_eval_iteration_period = model_eval_iteration_period
        self.variational_distribution_obj = variational_distribution_obj
        self.resample = resample
        self.n_sims = 0
        self.variational_params = None
        self.f_mat = None
        self.h_mat = None
        self.log_variational_mat = None
        self.grad_params_log_variational_mat = None
        self.variational_params_list = []
        self.log_posterior_unnormalized = None
        self.prior_obj_list = []
        self.samples_list = []
        self.parameter_list = []
        self.log_posterior_unnormalized_list = []
        self.ess_list = []
        self.noise_list = []
        self.elbo_list = []
        self.weights_list = []
        self.n_sims_list = []

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """Create iterator from problem description.

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator (obj): BBVI object
        """
        if iterator_name is None:
            method_options = config['method']['method_options']
        else:
            method_options = config[iterator_name]['method_options']
        if model is None:
            model_name = method_options['model']
            model = from_config_create_model(model_name, config)

        result_description = method_options.get('result_description', None)
        global_settings = config.get('global_settings', None)

        db = DB_module.database
        experiment_name = config['global_settings']['experiment_name']

        # set-up of the variational distribution
        variational_distribution_description = method_options.get("variational_distribution")
        # TODO this may have to be changed in the future
        num_variables = len(model.variables[0].variables)
        variational_distribution_description.update({"dimension": num_variables})
        variational_distribution_obj = variational_inference_utils.create_variational_distribution(
            variational_distribution_description
        )
        variational_family = variational_distribution_description.get("variational_family")
        n_samples_per_iter = method_options.get("n_samples_per_iter")
        variational_transformation = method_options.get("variational_transformation")
        random_seed = method_options.get("random_seed")
        max_feval = method_options.get("max_feval")
        variational_params_initialization_approach = method_options.get(
            "variational_parameter_initialization", None
        )
        memory = method_options.get("memory", 0)

        vis.from_config_create(config)

        optimization_options = method_options.get("optimization_options")
        natural_gradient_bool = optimization_options.get("natural_gradient", True)
        fim_dampening_bool = optimization_options.get("FIM_dampening", True)
        fim_decay_start_iter = optimization_options.get("decay_start_iteration", 50)
        fim_dampening_coefficient = optimization_options.get("dampening_coefficient", 1e-2)
        fim_dampening_lower_bound = optimization_options.get("FIM_dampening_lower_bound", 1e-8)
        variational_params_initialization_approach = method_options.get(
            "variational_parameter_initialization", None
        )

        export_quantities_over_iter = result_description.get("export_iteration_data")
        control_variates_scaling_type = method_options.get("control_variates_scaling_type")
        loo_cv_bool = method_options.get("loo_control_variates_scaling")
        if memory:
            model_eval_iteration_period = method_options.get("model_eval_iteration_period", 1000)
        else:
            model_eval_iteration_period = 1

        stochastic_optimizer = StochasticOptimizer.from_config_create_optimizer(
            method_options, "optimization_options"
        )
        resample = method_options.get("resample", False)
        return cls(
            global_settings=global_settings,
            model=model,
            result_description=result_description,
            db=db,
            experiment_name=experiment_name,
            variational_family=variational_family,
            variational_params_initialization_approach=variational_params_initialization_approach,
            n_samples_per_iter=n_samples_per_iter,
            variational_transformation=variational_transformation,
            random_seed=random_seed,
            max_feval=max_feval,
            num_variables=num_variables,
            memory=memory,
            natural_gradient_bool=natural_gradient_bool,
            fim_dampening_bool=fim_dampening_bool,
            fim_decay_start_iter=fim_decay_start_iter,
            fim_dampening_coefficient=fim_dampening_coefficient,
            fim_dampening_lower_bound=fim_dampening_lower_bound,
            export_quantities_over_iter=export_quantities_over_iter,
            control_variates_scaling_type=control_variates_scaling_type,
            loo_cv_bool=loo_cv_bool,
            variational_distribution_obj=variational_distribution_obj,
            stochastic_optimizer=stochastic_optimizer,
            model_eval_iteration_period=model_eval_iteration_period,
            resample=resample,
        )

    def core_run(self):
        """Core run for black-box variational inference."""
        _logger.info('Starting black box Bayesian variational inference...')
        start = time.time()
        # --------------------------------------------------------------------
        # -------- here comes the bbvi algorithm -----------------------------
        for _ in self.stochastic_optimizer:
            self._catch_non_converging_simulations()
            # Just to avoid constant spamming
            if self.stochastic_optimizer.iteration % 10 == 0:
                self._verbose_output()
                self._write_results()

            # Stop the optimizer in case of too many simulations
            if self.n_sims > self.max_feval:
                break

            self.variational_params_list.append(self.variational_params.copy())
            self._clearing_and_plots()
        # --------- end of bbvi algorithm ------------------------------------
        # --------------------------------------------------------------------
        end = time.time()

        if self.n_sims > self.max_feval:
            _logger.warning(f"Maximum probabilistic model calls reached")
        elif np.any(np.isnan(self.stochastic_optimizer.rel_L2_change)):
            _logger.warning(f"NaN(s) in the relative change of variational parameters")
        else:
            _logger.info(f"Finished sucessfully! :-)")
        _logger.info(f"Black box variational inference took {end-start} seconds.")
        _logger.info(f"---------------------------------------------------------")
        _logger.info(f"Cost of the analysis: {self.n_sims} simulation runs")
        _logger.info(f"Final ELBO: {self.elbo_list[-1]}")
        _logger.info(f"---------------------------------------------------------")
        _logger.info(f"Post run: Finishing, saving and cleaning...")

    def _catch_non_converging_simulations(self):
        """Reset variational parameters in case of failed simulations."""
        if np.isnan(self.stochastic_optimizer.rel_L2_change):
            self.variational_params = self.variational_params_list[:, -2]
            self.variational_distribution_obj.update_distribution_params(self.variational_params)

    def initialize_run(self):
        """Initialize the prior model and variational parameters."""
        _logger.info("Initialize Optimization run.")
        self._initialize_prior_model()
        self._initialize_variational_params()

        # set the gradient according to input
        self.stochastic_optimizer.gradient = self._get_gradient_function()
        self.stochastic_optimizer.current_variational_parameters = self.variational_params.reshape(
            -1, 1
        )

    def post_run(self):
        """Write results and potentially visualize them."""
        if self.result_description["write_results"]:
            result_dict = self._prepare_result_description()
            write_results(
                result_dict,
                self.global_settings["output_dir"],
                self.global_settings["experiment_name"],
            )
        vis.vi_visualization_instance.save_plots()

    def eval_model(self):
        """Evaluate model for the sample batch.

        Returns:
           result_dict (dict): Dictionary containing model response for sample batch
        """
        result_dict = self.model.evaluate()
        return result_dict

    def _initialize_prior_model(self):
        """Initialize the prior model."""
        # Read design of prior
        input_dict = self.model.get_parameter()
        # get random variables
        random_variables = input_dict.get('random_variables', None)
        # get random fields
        random_fields = input_dict.get('random_fields', None)

        if random_fields:
            raise NotImplementedError(
                'Variational inference for random fields is not yet implemented! Abort...'
            )
        # for each rv, evaluate prior all corresponding dims in current sample batch (column of
        # mat corresponds to all realization for this variable)
        for dim, rv_options in enumerate(random_variables.values()):
            self.prior_obj_list.append(mcmc_utils.create_proposal_distribution(rv_options))

    def eval_log_likelihood(self, sample_batch):
        """Calculate the log-likelihood of the observation data.

        Evaluation of the likelihood model for all inputs of the sample batch will trigger
        the actual forward simulation.

        Args:
            sample_batch (np.array): Sample-batch with samples row-wise

        Returns:
            log_likelihood (np.array): Vector of the log-likelihood function for all input
                                       samples of the current batch
        """
        # The first samples belong to simulation input
        # get simulation output (run actual forward problem)--> data is saved to DB
        self.model.update_model_from_sample_batch(sample_batch)
        log_likelihood = self.eval_model()
        self.noise_list.append(self.model.noise_var)

        return log_likelihood.flatten()

    def get_log_prior(self, sample_batch):
        """Evaluate the log prior of the model for a sample batch.

        The samples are transformed according to the selected
        transformation.

        Args:
            sample_batch (np.array): Sample batch for which the model should be evaluated

        Returns:
            log_prior (np.array): log-prior vector evaluated for current sample batch
        """
        log_prior_vec = np.zeros((self.n_samples_per_iter, 1))
        for dim, prior_distr in enumerate(self.prior_obj_list):
            # TODO Future proof this method
            log_prior_vec = log_prior_vec + prior_distr.logpdf(sample_batch[:, dim]).reshape(-1, 1)
        return log_prior_vec.flatten()

    def get_log_posterior_unnormalized(self, sample_batch):
        """Calculate the unnormalized log posterior for a sample batch.

        Args:
            sample_batch (np.array): Sample batch for which the model should be evaluated

        Returns:
            unnormalized_log_posterior (np.array): Values of unnormalized log posterior
                                                   distribution at positions of sample batch
        """
        # Transform the samples
        sample_batch = self._transform_samples(sample_batch)
        log_prior = self.get_log_prior(sample_batch)
        log_likelihood = self.eval_log_likelihood(sample_batch)
        log_posterior_unnormalized = log_likelihood + log_prior
        return log_posterior_unnormalized.flatten()

    def _verbose_output(self):
        """Give some informative outputs during the BBVI iterations."""
        mean_change = self.stochastic_optimizer.rel_L2_change * 100
        _logger.info("-" * 80)
        _logger.info(f"Iteration {self.stochastic_optimizer.iteration + 1} of BBVI algorithm")
        _logger.info(f"So far {self.n_sims} simulation runs")
        _logger.info(f"The elbo is: {self.elbo_list[-1]:.2f}")
        if self.memory > 0 and self.stochastic_optimizer.iteration > 0:
            _logger.info(
                f"ESS: {self.ess_list[-1]:.2f} of {(self.memory + 1) * self.n_samples_per_iter}"
            )
        if self.stochastic_optimizer.iteration > 1:
            rel_noise = (
                np.mean(np.abs(self.noise_list[-2] - self.noise_list[-1]) / self.noise_list[-2])
                * 100
            )
            _logger.info(
                f"Likelihood noise variance: {self.noise_list[-1]} (mean relative change "
                f"{rel_noise:.2f}) %"
            )
        _logger.info(f"L2 change of all variational parameters: " f"" f"{mean_change:.4f} %")

        # Avoids a busy screen
        if self.variational_params.shape[0] > 24:
            _logger.info(
                f"First 24 of {self.variational_params.shape[0]} variational parameters : \n"
            )
            _logger.info(self.variational_params[:24])
        else:
            _logger.info(f"Values of variational parameters: \n")
            _logger.info(self.variational_params)
        _logger.info("-" * 80)

    def _write_results(self):
        if self.result_description["write_results"]:
            result_dict = self._prepare_result_description()
            write_results(
                result_dict,
                self.global_settings["output_dir"],
                self.global_settings["experiment_name"],
            )

    def _initialize_variational_params(self):
        """Initialize the variational parameters.

        There are two possibilities:
            1. Random initialization:
                Is handeled by the variational distribution object
            2. Initialization based on the prior modeling (only for normal distributions!)
                Extract the prior moments and initialize the parameters based on them
        """
        if self.variational_params_initialization_approach == "random":
            self.variational_params = (
                self.variational_distribution_obj.initialize_parameters_randomly()
            )
        elif self.variational_params_initialization_approach == "prior":
            if self.variational_family == "normal":
                input_dict = self.model.get_parameter()
                random_variables = input_dict.get('random_variables', None)
                mu, cov = self._initialize_variational_params_from_prior(random_variables)
                var_params = self.variational_distribution_obj.construct_variational_params(mu, cov)
                self.variational_params = var_params
            else:
                raise ValueError(
                    f"Initializing the variational parameters based on the prior is only possible"
                    f"for distribution family 'normal'"
                )
        else:
            valid_initialization_types = {"random", "prior"}
            raise NotImplementedError(
                f"{self.variational_params_initialization_approach} is not known.\n"
                f"Valid options are {valid_initialization_types}"
            )

    def _initialize_variational_params_from_prior(self, random_variables):
        """Initialize the variational parameters based on the prior.

        The variational distribution might be transformed in a
        second step such that the actual variational distribution is of a
        different family. Only is used for normal distributions.

        Args:
            random_variables (dict): Dictionary containing the prior probabilistic description of
            the input variables
        """
        # Get the first and second moments of the prior distributions
        mean_list_prior = []
        std_list_prior = []
        if random_variables:
            for rv_options in random_variables.values():
                params = rv_options["distribution_parameter"]
                if rv_options["distribution"] == "normal":
                    mean_list_prior.append(params[0])
                    std_list_prior.append(params[1])
                elif rv_options["distribution"] == "lognormal":
                    mean_list_prior.append(np.exp(params[0] + (params[1] ** 2) / 2))
                    std_list_prior.append(mean_list_prior[-1] * np.sqrt(np.exp(params[1] ** 2) - 1))
                elif rv_options["distribution"] == "uniform":
                    mean_list_prior.append((params[1] - params[0]) / 2)
                    std_list_prior.append((params[1] - params[0]) / np.sqrt(12))
                else:
                    distr_type = rv_options["distribution"]
                    raise NotImplementedError(
                        f"Your prior type {distr_type} is not supported in"
                        f" the variational approach! Abort..."
                    )

        # Set the mean and std-deviation params of the variational distr such that the
        # transformed distribution would match the moments of the prior
        if self.variational_transformation == 'exp':
            mean_list_variational = [
                np.log(E ** 2 / np.sqrt(E ** 2 + S ** 2))
                for E, S in zip(mean_list_prior, std_list_prior)
            ]
            std_list_variational = [
                np.sqrt(np.log(1 + S ** 2 / E ** 2))
                for E, S in zip(mean_list_prior, std_list_prior)
            ]
        elif self.variational_transformation is None:
            mean_list_variational = mean_list_prior
            std_list_variational = std_list_prior
        else:
            raise ValueError(
                f"The transformation type {self.variational_transformation} for the "
                f"variational density is unknown! Abort..."
            )

        return np.array(mean_list_variational), np.diag(std_list_variational) ** 2

    def _transform_samples(self, x_mat):
        """Transform samples.

        Transform samples of the variational distribution according to the specified
        transformation mapping.

        Args:
            x_mat (np.array): Samples of the variational distribution

        Returns:
            x_mat_trans (np.array): Transformed samples of variational distribution
        """
        if self.variational_transformation == 'exp':
            x_mat_trans = np.exp(x_mat)
        elif self.variational_transformation is None:
            x_mat_trans = x_mat
        else:
            raise ValueError(
                f"The transformation type {self.variational_transformation} for the "
                f"variational density is unknown! Abort..."
            )

        return x_mat_trans

    def _prepare_result_description(self):
        """Creates the dictionary for the result pickle file.

        Returns:
            result_description (dict): Dictionary with result summary of the analysis
        """
        result_description = {
            "final_elbo": self.elbo_list[-1],
            "batch_size": self.n_samples_per_iter,
            "number_of_sim": self.n_sims,
            "natural_gradient": self.natural_gradient_bool,
            "fim_dampening": self.fim_dampening_bool,
            "control_variates_scaling_type": self.control_variates_scaling_type,
            "loo_control_variates": self.loo_cv_bool,
            "variational_parameter_initialization": self.variational_params_initialization_approach,
        }

        if self.export_quantities_over_iter:
            result_description.update(
                {
                    "params_over_iter": self.variational_params_list,
                    "likelihood_noise_var": self.noise_list,
                    "elbo": self.elbo_list,
                }
            )
            if self.memory > 0:
                result_description.update(
                    {
                        "ESS": self.ess_list,
                        "memory": self.memory,
                    }
                )

        distribution_dict = self.variational_distribution_obj.export_dict(self.variational_params)
        if self.variational_transformation == "exp":
            distribution_dict.update(
                {"variational_transformation": self.variational_transformation}
            )

        result_description.update({"variational_distr": distribution_dict})
        return result_description

    def _averaged_control_variates_scalings(self, f_mat, h_mat):
        """Averaged control variate scalings.

        This function computes the control variate scaling averaged over the
        components of the control variate.

        Args:
            f_mat (np.array): Column-wise MC gradient samples
            h_mat (np.array): Column-wise control variate samples

        Returns:
            cv_scaling (np.array): Columnvector with control variate scalings
        """
        dim = len(h_mat)
        cov_sum = 0
        var_sum = 0

        for f_dim, h_dim in zip(f_mat, h_mat):
            cov_sum += np.cov(f_dim, h_dim)[0, 1]
            var_sum += np.var(h_dim)
        cv_scaling = np.ones((dim, 1)) * cov_sum / var_sum
        return cv_scaling

    def _componentwise_control_variates_scalings(self, f_mat, h_mat):
        """Computes the componentwise control variates scaling.

        I.e., every component of the control variate separately is computed seperately.

        Args:
            f_mat (np.array): Column-wise MC gradient samples
            h_mat (np.array): Column-wise control variate samples
        Returns:
            cv_scaling (np.array): Columnvector with control variate scalings
        """
        dim = len(h_mat)
        cv_scaling = np.ones((dim, 1))
        for i in range(dim):
            cv_scaling[i] = np.cov(f_mat[i], h_mat[i])[0, 1]
            cv_scaling[i] = cv_scaling[i] / np.var(h_mat[i])
        return cv_scaling

    def _loo_control_variates_scalings(self, cv_obj, f_mat, h_mat):
        """Leave one out control variates.

        To reduce bias in the MC and control variate saling estimation
        Ranganath proposed a leave-one-out procedure to estimate the control
        variate scalings. Each sample has its own scaling that is computed
        using f_mat and h_mat without the values related to itself. (see
        http://arks.princeton.edu/ark:/88435/dsp01pr76f608w) Is slow!

        Args:
            cv_obj (control variate function): A control variate scaling function
            f_mat (np.array): Column-wise MC gradient samples
            h_mat (np.array): Column-wise control variate samples

        Returns:
            cv_scaling (np.array): Colum-wise control variate scalings for the different samples
        """
        cv_scaling = []
        for i in range(f_mat.shape[1]):
            scv = cv_obj(np.delete(f_mat, i, 1), np.delete(h_mat, i, 1))
            cv_scaling.append(scv)
        cv_scaling = np.concatenate(cv_scaling, axis=1)
        return cv_scaling

    def _get_control_variates_scalings(self):
        """Calculate the control variate scalings.

        Returns:
            cv_scaling (np.array): Scaling for the control variate
        """
        if self.control_variates_scaling_type == "componentwise":
            cv_scaling_obj = self._componentwise_control_variates_scalings
        elif self.control_variates_scaling_type == "averaged":
            cv_scaling_obj = self._averaged_control_variates_scalings
        else:
            valid_options = {"componentwise", "averaged"}
            raise NotImplementedError(
                f"{self.control_variates_scaling_type} unknown, valid types are {valid_options}"
            )
        if self.loo_cv_bool:
            cv_scaling = self._loo_control_variates_scalings(cv_scaling_obj, self.f_mat, self.h_mat)
        else:
            cv_scaling = cv_scaling_obj(self.f_mat, self.h_mat)
        return cv_scaling

    def _get_gradient_function(self):
        """Select the gradient function for the stochastic optimizer.

        Two options exist, with or without natural gradient.

        Returns:
            obj: function to evaluate the gradient
        """
        if self.natural_gradient_bool:
            FIM = self._get_fim()
            gradient = lambda variational_parameters: np.linalg.solve(
                self._get_fim(), self._calculate_elbo_gradient(variational_parameters)
            )
        else:
            gradient = lambda variational_parameters: self._calculate_elbo_gradient(
                variational_parameters
            )
        return gradient

    def _calculate_elbo_gradient(self, variational_parameters):
        """Estimate the ELBO gradient expression.

        Based on MC with importance sampling with the samples of previous iterations if desired.
        The score function is used as a control variate. No Rao-Blackwellization scheme
        is used.

        Returns:
            elbo gradient as column vector (np.array)
        """
        self.variational_params = variational_parameters.flatten()

        # Check if evaluating the probabilistic model is necessary
        self._check_if_sampling_necessary()
        if self.sampling_bool:
            self._sample_gradient_from_probabilistic_model()

        # Use IS sampling (if enabled)
        selfnormalized_weights_is, normalizing_constant_is = self._prepare_importance_sampling()
        if self.stochastic_optimizer.iteration > self.memory and self.memory > 0 and self.resample:
            # Number of samples to resample currently set to the max achievable ESS
            n_samples = int(self.n_samples_per_iter * self.memory)
            # Resample
            self._resample(selfnormalized_weights_is, n_samples)
            # After resampling the weights
            selfnormalized_weights_is, normalizing_constant_is = 1 / n_samples, n_samples

        #  Evaluate the logpdf and grad params logpdf function of the variational distribution
        self._evaluate_variational_distribution_for_batch()
        self._filter_failed_simulations()

        # Compute the MC samples, without control variates but with IS weights
        self.f_mat = (
            selfnormalized_weights_is
            * self.grad_params_log_variational_mat
            * (self.log_posterior_unnormalized - self.log_variational_mat)
        )

        # Compute the control variate at the given samples
        self.h_mat = selfnormalized_weights_is * self.grad_params_log_variational_mat

        # Get control variate scalings
        a = self._get_control_variates_scalings()

        # MC gradient estimation with control variates
        self.grad_elbo = normalizing_constant_is * np.mean(
            self.f_mat - a * self.h_mat, axis=1
        ).reshape(-1, 1)

        # Compute the logpdf for the elbo estimate (here no IS is used)
        self._calculate_elbo(selfnormalized_weights_is, normalizing_constant_is)

        self.n_sims_list.append(self.n_sims)

        # Avoid NaN in the elbo gradient
        return np.nan_to_num(self.grad_elbo)

    def _sample_gradient_from_probabilistic_model(self):
        """Evaluate probabilistic model."""
        # Increase model call counter
        n_samples = self.n_samples_per_iter
        self.n_sims += n_samples
        # Draw samples for the current iteration
        self.samples = self.variational_distribution_obj.draw(self.variational_params, n_samples)

        # Calls the (unnormalized) probabilistic model
        self.log_posterior_unnormalized = self.get_log_posterior_unnormalized(self.samples)

    def _evaluate_variational_distribution_for_batch(self):
        """Evaluate logpdf and score function."""
        self.log_variational_mat = self.variational_distribution_obj.logpdf(
            self.variational_params, self.samples
        )

        self.grad_params_log_variational_mat = self.variational_distribution_obj.grad_params_logpdf(
            self.variational_params, self.samples
        )

        # Convert if NaNs to floats. For high dimensional RV floating point issues
        # might be avoided this way
        self.log_variational_mat = np.nan_to_num(self.log_variational_mat)
        self.grad_params_log_variational_mat = np.nan_to_num(self.grad_params_log_variational_mat)

    def _resample(self, selfnormalized_weights, n_samples):
        """Stratified resampling.

        Args:
            selfnormalized_weights (np.array): weights of the samples
            n_samples (int): number of samples
        """
        random_sample_within_bins = (np.random.rand(n_samples) + np.arange(n_samples)) / n_samples

        idx = []
        cumulative_probability = np.cumsum(selfnormalized_weights)
        i, j = 0, 0
        while i < n_samples:
            if random_sample_within_bins[i] < cumulative_probability[j]:
                idx.append(j)
                i += 1
            else:
                j += 1

        self.log_posterior_unnormalized = self.log_posterior_unnormalized[idx]
        self.samples = np.array([self.samples[j] for j in idx])

    def _check_if_sampling_necessary(self):
        """Check if resampling is necessary.

        Resampling is necessary if on of the following condition is True:
            1. No memory is used
            2. Not enought samples in memory
            3. ESS number is too low
            4. Every model_eval_iteration_period
        """
        max_ess = self.n_samples_per_iter * (self.memory)
        self.sampling_bool = (
            self.memory == 0
            or self.stochastic_optimizer.iteration <= self.memory
            or self.ess_list[-1] < 0.5 * max_ess
            or self.stochastic_optimizer.iteration % self.model_eval_iteration_period == 0
        )

    def _calculate_elbo(self, selfnormalized_weights, normalizing_constant):
        """Calculate the ELBO.

        Args:
            selfnormalized_weights (np.array): selfnormalized importance sampling weights
            normalizing_constant (int): Importance sampling normalizing constant
        """
        instant_elbo = selfnormalized_weights * (
            self.log_posterior_unnormalized - self.log_variational_mat
        )
        self.elbo_list.append(normalizing_constant * np.mean(instant_elbo))

    def _prepare_importance_sampling(self):
        r"""Helper functions for the importance sampling.

        Importance sampling based gradient computation (if enabled). This includes:
            1. Store samples, variational parameters and probabilistic model evaluations
            2. Update variables samples and log_posterior_unnormalized
            3. Compute autonormalized weights and the normalizing constant

        The normalizing constant is a constant in order to recover the proper weights values. The
        gradient estimation is multiplied with this constant in order to avoid a bias. This can
        be done since for any constant :math:`a`:
        :math:`\int_x h(x) p(x) dx = a \int_x h(x) \frac{1}{a}p(x) dx`

        Returns:
            selfnormalized_weights (np.array): Row-vector with selfnormalized weights
            normalizing_constant (int): Normalizing constant
            samples (np.array): Eventually extended row-wise samples
        """
        # Values if no IS is used or for the first iteration
        selfnormalized_weights = 1
        normalizing_constant = 1

        # If importance sampling is used
        if self.memory > 0:
            self._update_sample_and_posterior_lists()

            # The number of iterations that we want to keep the samples and model evals
            if self.stochastic_optimizer.iteration > 0:
                weights_is = self.get_importance_sampling_weights(self.parameter_list, self.samples)

                # Self normalize weighs
                normalizing_constant = np.sum(weights_is)
                selfnormalized_weights = weights_is / normalizing_constant
                self.ess_list.append(1 / np.sum(selfnormalized_weights ** 2))
                self.weights_list.append(weights_is)

        return selfnormalized_weights, normalizing_constant

    def _update_sample_and_posterior_lists(self):
        """Assemble the samples for IS MC gradient estimation."""
        # Check if probabilistic model was sampled
        if self.sampling_bool:
            # Store the current samples, parameters and probabilistic model evals
            self.parameter_list.append(self.variational_params)
            self.samples_list.append(self.samples)
            self.log_posterior_unnormalized_list.append(self.log_posterior_unnormalized)

        # The number of iterations that we want to keep the samples and model evals
        if self.stochastic_optimizer.iteration >= self.memory:
            self.parameter_list = self.parameter_list[-(self.memory + 1) :]
            self.samples_list = self.samples_list[-(self.memory + 1) :]
            self.log_posterior_unnormalized_list = self.log_posterior_unnormalized_list[
                -(self.memory + 1) :
            ]

            self.samples = np.concatenate(self.samples_list, axis=0)
            self.log_posterior_unnormalized = np.concatenate(
                self.log_posterior_unnormalized_list, axis=0
            )

    def get_importance_sampling_weights(self, variational_params_list, samples):
        r"""Get the importance sampling weights for the MC gradient estimation.

        Uses a special computation of the weights using the logpdfs to reduce
        numerical issues:

        :math: `w=\frac{q_i}{\sum_{j=0}^{memory+1} \frac{1}{memory+1}q_j}=\frac{(memory +1)}
        {1+\sum_{j=0}^{memory}exp(lnq_j-lnq_i)}`

        and is therefore slightly slower.

        Args:
            variational_params_list (list): variational parameters list of the current and the
                                            desired previous iterations
            samples (np.array): Row-wise samples for the MC gradient estimation

        Returns:
            weights (np.array): (Unnormalized) weights for the ISMC evaluated for the given samples
        """
        weights = 1
        n_mixture = len(variational_params_list)
        log_pdf_current_iteration = self.variational_distribution_obj.logpdf(
            self.variational_params, samples
        )
        for j in range(0, n_mixture - 1):
            weights += np.exp(
                self.variational_distribution_obj.logpdf(variational_params_list[j], samples)
                - log_pdf_current_iteration
            )
        weights = n_mixture / weights
        return weights

    def evaluate_variational_distribution_for_batch(self, samples):
        """Evaluate logpdf and score function of the variational distribution.

        Args:
            samples (np.array): Row-wise samples
        """
        self.log_variational_mat = self.variational_distribution_obj.logpdf(
            self.variational_params, samples
        )

        self.grad_params_log_variational_mat = self.variational_distribution_obj.grad_params_logpdf(
            self.variational_params, samples
        )

        # Convert if NaNs to floats. For high dimensional RV floating point issues
        # might be avoided this way
        self.log_variational_mat = np.nan_to_num(self.log_variational_mat)
        self.grad_params_log_variational_mat = np.nan_to_num(self.grad_params_log_variational_mat)

    def _filter_failed_simulations(self):
        """Filter samples failed simulations."""
        # Indices where the log joint is a nan
        idx = np.where(~np.isnan(self.log_posterior_unnormalized))[0]
        if len(idx) != len(self.log_posterior_unnormalized):
            _logger.warning(f"At least one probabilistic model call resulted in a NaN")
        if self.log_variational_mat.ndim > 1:
            self.log_variational_mat = self.log_variational_mat[:, idx]
        else:
            self.log_variational_mat = self.log_variational_mat[idx]
        self.grad_params_log_variational_mat = self.grad_params_log_variational_mat[:, idx]
        self.log_posterior_unnormalized = self.log_posterior_unnormalized[idx]

    def _clearing_and_plots(self):
        """Visualization and clear some internal variables."""
        # clear internal variables
        self.log_variational_mat = None
        self.log_posterior_unnormalized = None
        self.grad_params_log_variational_mat = None

        # some plotting and output
        vis.vi_visualization_instance.plot_convergence(
            self.stochastic_optimizer.iteration,
            self.variational_params_list,
            self.elbo_list,
        )

    def _get_fim(self):
        """Get the FIM for the current variational distribution.

        Add dampening if desired.

        Returns:
            fisher (np.array): fisher information matrix of the variational distribution
        """
        FIM = self.variational_distribution_obj.fisher_information_matrix(self.variational_params)
        if self.fim_dampening_bool:
            if self.stochastic_optimizer.iteration > self.fim_decay_start_iter:
                dampening_coefficient = self.fim_dampening_coefficient * np.exp(
                    -(self.stochastic_optimizer.iteration - self.fim_decay_start_iter)
                    / self.fim_decay_start_iter
                )
                dampening_coefficient = max(self.fim_dampening_lower_bound, dampening_coefficient)
            else:
                dampening_coefficient = self.fim_dampening_coefficient
            FIM = FIM + np.eye(len(FIM)) * dampening_coefficient
        return FIM
