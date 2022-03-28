import logging
import pprint
import time

import autograd.numpy as np
import numpy as npy
from autograd import grad
from scipy.stats import multivariate_normal as mvn

import pqueens.database.database as DB_module
import pqueens.visualization.variational_inference_visualization as vis
from pqueens.external_geometry import from_config_create_external_geometry
from pqueens.iterators.iterator import Iterator
from pqueens.models import from_config_create_model
from pqueens.utils import mcmc_utils, variational_inference_utils
from pqueens.utils.process_outputs import write_results

_logger = logging.getLogger(__name__)


class VIRPIterator(Iterator):
    """Variational Inference with reparameterization trick (VIRP) iterator for
    Bayesian inverse problems. This variational inference approach requires
    model gradients/jacobians w.r.t. the parameters/the parameterization of the
    inverse problem. The latter can be provided by:

        - A finite differences approximation of the gradient/jacobian, which requires in the
          simplest case d+1 additional solver calls
        - An externally provided gradient/jacobian that was, e.g., calculated via adjoint methods or
          automated differentiation

    The mathematical details of the algorithm can be found in [1], [2], [3]

    References:
        [1]: Kingma, D. P., Salimans, T., & Welling, M. (2015). Variational dropout and the local
            reparameterization trick. Advances in neural information processing systems,
            28, 2575-2583.
        [2]: Roeder, G., Wu, Y., & Duvenaud, D. (2017). Sticking the landing: Simple,
             lower-variance
             gradient estimators for variational inference. arXiv preprint arXiv:1703.09194.
        [3]: Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A
             review for statisticians. Journal of the American statistical Association, 112(518),
             859-877.

    Attributes:
        result_description (dict): Settings for storing and visualizing the results
        db (obj): Database object
        experiment_name (str): Name of the current QUEENS experiment
        variational_samples (np.array): Samples of the variational distribution
        variational_params (np.array): Array of parameter values of the variational
                                      distribution
        variational_distribution_obj (obj): Instance/object of a (variational) distribution class
        relative_change_variational_params (float): Relative change of the variational parameters
                                                    inbetween iterations
        min_requ_relative_change_variational_params (float): Minimal required relative change of the
                                                             variational parameters for convergence
        learning_rate (float): Adams optimizer learning rate
        n_samples_per_iter (int): Number of Monte-Carlo samples (to approximate expectation in VI)
                                  per iteration
        iteration_num (int): Iteration number
        log_likelihood_vec (np.array): Vector with likelihood score per sample
        elbo_lst (lst): List with values of the evidence lower bound per iteration
        log_posterior_unnormalized (float): Value of unnormalized log posterior for given model
                                            output
        v_param_adams (float): Parameter of stochastic ascent Adam optimizer
        m_param_adams (float): Parameter of stochastic ascent Adam optimizer
        random_seed (int): Random seed for random number generation
        prior_obj_list (lst): List containing the (intependent) prior distribution objects
        max_feval (int): Maximum number of solver calls / function evaluations
        num_variables (int): Number of random variables
        geometry_obj (obj): Instance of external geometry class
        variational_transformation (str): String indicating the type of a potential transformation
                                          of the variational distribution
        variational_family (str): String that indicates the type/family of the variational
                                  distribution
        variational_params_initialization_approach (str): Flag to decide how to initialize the
                                                          variational paramaters
        num_iter_average_convergence (float): Number of iteration used in assesing convergence
        variational_params_array (np.array): Column-wise parameters from first to last iteration
        n_sims (int): Number of probabilistic model calls
        export_quantities_over_iter (boolean): True if data (variational_params, elbo) should
                                               be exported in the pickle file
        noise_lst (list): Gaussian likelihood noise variance values.
        clipping_bool (boolean): True if gradient clipping should be used
        gradient_clipping_norm_threshold (float): Threshold for gradient clipping
        natural_gradient_bool (boolean): True if natural gradient should be used
        fim_dampening_bool (boolean): True if FIM dampening should be used
        fim_decay_start_iter (float): Iteration at which the FIM dampening is started
        fim_dampening_coefficient (float): Initial nugget term value for the FIM dampening
        fim_dampening_lower_bound (float): Lower bound on the FIM dampening coefficient
        score_function_bool (bool): Boolean flag to decide wheater the score function term
                                    should be considered in the elbo gradient. If true the
                                    score function is considered.
        finite_difference_step (float): Finite difference step size

    Returns:
        virp_obj (obj): Instance of the VIRPIterator
    """

    def __init__(
        self,
        model,
        global_settings,
        result_description,
        db,
        experiment_name,
        min_requ_relative_change_variational_params,
        learning_rate,
        n_samples_per_iter,
        random_seed,
        max_feval,
        num_variables,
        geometry_obj,
        variational_family,
        variational_approximation_type,
        iteration_num,
        variational_samples,
        variational_params,
        variational_distribution_obj,
        relative_change_variational_params_lst,
        log_likelihood_vec,
        elbo_lst,
        log_posterior_unnormalized,
        v_param_adams,
        m_param_adams,
        prior_obj_list,
        variational_transformation,
        variational_params_initialization_approach,
        num_iter_average_convergence,
        variational_params_array,
        n_sims,
        export_quantities_over_iter,
        noise_lst,
        clipping_bool,
        gradient_clipping_norm_threshold,
        natural_gradient_bool,
        fim_dampening_bool,
        fim_decay_start_iter,
        fim_dampening_coefficient,
        fim_dampening_lower_bound,
        score_function_bool,
        finite_difference_step,
    ):
        super().__init__(model, global_settings)

        self.result_description = result_description
        self.db = db
        self.experiment_name = experiment_name
        self.variational_samples = variational_samples
        self.variational_params = variational_params
        self.variational_distribution_obj = variational_distribution_obj
        self.relative_change_variational_params = relative_change_variational_params_lst
        self.min_requ_relative_change_variational_params = (
            min_requ_relative_change_variational_params
        )
        self.learning_rate = learning_rate
        self.n_samples_per_iter = n_samples_per_iter
        self.iteration_num = iteration_num
        self.log_likelihood_vec = log_likelihood_vec
        self.elbo_lst = elbo_lst
        self.log_posterior_unnormalized = log_posterior_unnormalized
        self.v_param_adams = v_param_adams
        self.m_param_adams = m_param_adams
        self.random_seed = random_seed
        self.prior_obj_list = prior_obj_list
        self.max_feval = max_feval
        self.num_variables = num_variables
        self.geometry_obj = geometry_obj
        self.variational_transformation = variational_transformation
        self.variational_family = variational_family
        self.variational_approximation_type = variational_approximation_type
        self.variational_params_initialization_approach = variational_params_initialization_approach
        self.num_iter_average_convergence = num_iter_average_convergence
        self.variational_params_array = variational_params_array
        self.n_sims = n_sims
        self.export_quantities_over_iter = export_quantities_over_iter
        self.noise_lst = noise_lst
        self.clipping_bool = clipping_bool
        self.gradient_clipping_norm_threshold = gradient_clipping_norm_threshold
        self.natural_gradient_bool = natural_gradient_bool
        self.fim_decay_start_iter = fim_decay_start_iter
        self.fim_dampening_coefficient = fim_dampening_coefficient
        self.fim_dampening_lower_bound = fim_dampening_lower_bound
        self.fim_dampening_bool = fim_dampening_bool
        self.score_function_bool = score_function_bool
        self.finite_difference_step = finite_difference_step

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """Create variational inference reparameterization trick iterator from
        problem description.

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator (optional)
            model (model): Model to use (optional)

        Returns:
            iterator (obj): VIRP object
        """
        if iterator_name is None:
            method_options = config['method']['method_options']
        else:
            method_options = config[iterator_name]['method_options']
        if model is None:
            model_name = method_options['model']
            model = from_config_create_model(model_name, config)

        result_description = method_options.get('result_description', None)
        export_quantities_over_iter = result_description.get("export_iteration_data")
        global_settings = config.get('global_settings', None)

        db = DB_module.database
        experiment_name = config['global_settings']['experiment_name']
        relative_change_variational_params = method_options.get(
            'min_relative_change_variational_params', 0.01
        )
        learning_rate = method_options.get("learning_rate")
        n_samples_per_iter = method_options.get("n_samples_per_iter")
        random_seed = method_options.get("random_seed")
        max_feval = method_options.get("max_feval")
        num_variables = len(model.variables[0].variables)
        geometry_obj = from_config_create_external_geometry(config)
        num_iter_average_convergence = method_options.get("num_iter_average_convergence", "5")
        score_function_bool = method_options.get("score_function_bool", True)
        finite_difference_step = method_options.get("finite_difference_step")

        # natural gradients and clipping
        optimization_options = method_options.get("optimization_options")
        clipping_bool = optimization_options.get("gradient_clipping")
        gradient_clipping_norm_threshold = optimization_options.get(
            "gradient_clipping_norm_threshold", 1e6
        )
        natural_gradient_bool = optimization_options.get("natural_gradient", True)
        learning_rate = optimization_options.get("learning_rate")
        fim_dampening_bool = optimization_options.get("FIM_dampening", True)
        fim_decay_start_iter = optimization_options.get("decay_start_iteration", 50)
        fim_dampening_coefficient = optimization_options.get("dampening_coefficient", 1e-2)
        fim_dampening_lower_bound = optimization_options.get("FIM_dampening_lower_bound", 1e-8)

        # set-up of the variational distribution
        variational_distribution_description = method_options.get("variational_distribution")
        # TODO this may have to be changed in the future
        variational_distribution_description.update({"dimension": num_variables})
        variational_distribution_obj = variational_inference_utils.create_variational_distribution(
            variational_distribution_description
        )
        variational_transformation = method_options.get("variational_transformation")
        variational_params_initialization_approach = method_options.get(
            "variational_parameter_initialization", None
        )
        variational_family = variational_distribution_description.get("variational_family")
        variational_approximation_type = variational_distribution_description.get(
            "variational_approximation_type"
        )

        # Some initializations
        iteration_num = 0
        variational_samples = None
        variational_params = None
        relative_change_variational_params_lst = []
        log_likelihood_vec = None
        elbo_lst = []
        log_posterior_unnormalized = None
        v_param_adams = 0
        m_param_adams = 0
        n_sims = 0
        prior_obj_list = []
        variational_params_array = None
        noise_lst = []

        # Initialize visualization module
        vis.from_config_create(config)

        return cls(
            global_settings=global_settings,
            model=model,
            result_description=result_description,
            db=db,
            experiment_name=experiment_name,
            min_requ_relative_change_variational_params=relative_change_variational_params,
            learning_rate=learning_rate,
            n_samples_per_iter=n_samples_per_iter,
            random_seed=random_seed,
            max_feval=max_feval,
            num_variables=num_variables,
            geometry_obj=geometry_obj,
            variational_family=variational_family,
            variational_approximation_type=variational_approximation_type,
            iteration_num=iteration_num,
            variational_samples=variational_samples,
            variational_params=variational_params,
            variational_distribution_obj=variational_distribution_obj,
            relative_change_variational_params_lst=relative_change_variational_params_lst,
            log_likelihood_vec=log_likelihood_vec,
            elbo_lst=elbo_lst,
            log_posterior_unnormalized=log_posterior_unnormalized,
            v_param_adams=v_param_adams,
            m_param_adams=m_param_adams,
            prior_obj_list=prior_obj_list,
            variational_transformation=variational_transformation,
            variational_params_initialization_approach=variational_params_initialization_approach,
            num_iter_average_convergence=num_iter_average_convergence,
            variational_params_array=variational_params_array,
            n_sims=n_sims,
            export_quantities_over_iter=export_quantities_over_iter,
            noise_lst=noise_lst,
            clipping_bool=clipping_bool,
            gradient_clipping_norm_threshold=gradient_clipping_norm_threshold,
            natural_gradient_bool=natural_gradient_bool,
            fim_dampening_bool=fim_dampening_bool,
            fim_decay_start_iter=fim_decay_start_iter,
            fim_dampening_coefficient=fim_dampening_coefficient,
            fim_dampening_lower_bound=fim_dampening_lower_bound,
            score_function_bool=score_function_bool,
            finite_difference_step=finite_difference_step,
        )

    def initialize_run(self):
        """Initialize BBVI object with experimental data and the models for.

        - the underlying likelihood model
        - the variational distribution model
        - the underlying prior model

        Returns:
            None
        """

        _logger.info("Initialize Optimization run.")
        # check for random fields
        parameters = self.model.get_parameter()
        random_fields = parameters.get('random_fields', None)
        if random_fields:
            self._random_field_preprocessing(random_fields)

        self._initialize_prior_model()
        self._initialize_variational_params()
        self._get_gradient_objects()

    def _catch_non_converging_simulations(self):
        """Reset variational parameters in case of non-converging simulation
        runs.

        Returns:
            None
        """
        if len(self.relative_change_variational_params) > 0:
            if np.any(np.isnan(self.relative_change_variational_params[-1])):
                self.variational_params = self.variational_params_array[:, -2]
                self.variational_distribution_obj.update_distribution_params(
                    self.variational_params
                )

    def core_run(self):
        """Core run for variational inference with reparameterization trick.

        Retruns:
            None
        """

        # Some clarification of terms:
        # params: actual parameters for which we solve the inverse problem
        # sample: sample of the variational distribution
        # variational_params: variational parameters that parameterize var,.distr. of params
        #                     (lambda)

        _logger.info('Starting variational inference...')
        start = time.time()

        # --------------------------------------------------------------------
        # -------- here comes the algorithm ----------------------------------
        while (self._check_convergence()) and ((self.n_sims) <= self.max_feval):
            self._catch_non_converging_simulations()

            # generate the base samples (zero mean and unit variance)
            # and initialize elbo
            sample_batch = variational_inference_utils.draw_base_samples_from_standard_normal(
                self.n_samples_per_iter, self.num_variables
            )
            sample_elbo_grad = 0
            log_unnormalized_posterior = 0

            # iterate over samples in sample batch
            for sample in sample_batch:

                # Note the steps below are done for a batch of n samples
                variational_params = np.array(self.variational_params)
                num_var_params = variational_params.size
                mu_params = variational_params[: int(num_var_params / 2)]
                sigma_params = variational_params[int(num_var_params / 2) :]

                params = []
                grad_reparameterization = []

                # evaluate the reparameterized gradient of the model w.r.t to the parameters at the
                # sample location
                for (
                    num,
                    sample_dim,
                ) in enumerate(sample):
                    var_params_sample_dim = np.array([mu_params[num], sigma_params[num]])
                    params.append(
                        variational_inference_utils.conduct_reparameterization(
                            var_params_sample_dim, sample_dim
                        )
                    )
                    grad_reparameterization.append(
                        self.calculate_grad_reparameterization_variational_params(
                            var_params_sample_dim, sample_dim
                        )
                    )

                # evaluate the necessary contributions
                params = np.array(params)
                grad_reparameterization = np.array(grad_reparameterization).reshape(
                    -1, 1, order='F'
                )

                grad_log_prior = self.calculate_grad_log_prior_params(params)
                log_prior = self.get_log_prior(params)
                # pylint: disable=line-too-long
                grad_variational = (
                    variational_inference_utils.calculate_grad_log_variational_distr_params(
                        self.grad_log_variational_distr_params, params, variational_params
                    )
                )
                # pylint: enable=line-too-long

                # check if score function should be added to the derivative
                if self.score_function_bool:
                    # pylint: disable=line-too-long
                    # TODO the method does the same as variational_inference_utils.grad_params_logpdf but reuses pre-computed gradients here
                    grad_log_variational_distr_variational_params = variational_inference_utils.calculate_grad_log_variational_distr_variational_params(
                        grad_reparameterization, grad_variational
                    )
                    # pylint: enable=line-too-long

                grad_log_likelihood, log_likelihood = self.calculate_grad_log_likelihood_params(
                    params
                )

                # calculate the elbo gradient for one sample
                sample_elbo_grad += (
                    np.vstack(
                        (grad_log_likelihood.reshape(-1, 1), grad_log_likelihood.reshape(-1, 1))
                    )
                    + np.vstack((grad_log_prior.reshape(-1, 1), grad_log_prior.reshape(-1, 1)))
                    - np.vstack((grad_variational.reshape(-1, 1), grad_variational.reshape(-1, 1)))
                ) * grad_reparameterization.reshape(-1, 1)
                if self.score_function_bool:
                    sample_elbo_grad = (
                        sample_elbo_grad - grad_log_variational_distr_variational_params
                    )

                # calculate the unnormalized posterior for one sample
                log_unnormalized_posterior += log_likelihood + log_prior

            # MC estimate of elbo gradient
            grad_elbo = sample_elbo_grad / self.n_samples_per_iter
            # MC estimate for unnormalized posterior
            log_unnormalized_posterior = log_unnormalized_posterior / self.n_samples_per_iter

            self._update_variational_params(grad_elbo)
            self._calculate_elbo(log_unnormalized_posterior)
            self._plotting()
            self._verbose_output()
            self.iteration_num += 1
            # TODO adopt this in case of ajoints
            self.n_sims += self.n_samples_per_iter * (self.num_variables + 1)
        # --------- end of algorithm -----------------------------------------
        # --------------------------------------------------------------------

        end = time.time()

        _logger.info(f"Finished successfully! :-)")
        _logger.info(f"Variational inference took {end-start} seconds.")
        _logger.info(f"---------------------------------------------------------")
        _logger.info(f"Cost of the analysis: {self.n_sims} simulation runs")
        _logger.info(f"Final ELBO: {self.elbo_lst[-1]}")
        _logger.info(f"---------------------------------------------------------")
        _logger.info(f"Post run: Finishing, saving and cleaning...")

    def _check_convergence(self):
        """Check the convergence criterion for the BBVI iterator.

        Returns:
            convergence_bool (bool): True if not yet converged. False if converged
        """
        if self.iteration_num > self.num_iter_average_convergence:
            convergence_bool = (
                np.any(
                    np.mean(
                        self.relative_change_variational_params[
                            -self.num_iter_average_convergence :
                        ],
                        axis=0,
                    )
                    > self.min_requ_relative_change_variational_params
                )
            ) and not np.any(np.isnan(self.relative_change_variational_params[-1]))
        else:
            convergence_bool = True
        return convergence_bool

    def _plotting(self):
        """Pass data to the visualization object.

        Returns:
            None
        """
        # some plotting and output
        vis.vi_visualization_instance.plot_convergence(
            self.iteration_num,
            self.variational_params_array,
            self.elbo_lst,
            self.relative_change_variational_params,
        )

    def _random_field_preprocessing(self, random_fields):
        """Preprocessing for random fields, e.g., spectral representation or
        PCA representation.

        Returns:
            None
        """
        raise NotImplementedError("Random field preprocessing is not implemented, yet! Abort...")

    def _get_gradient_objects(self):
        """Get the automatic differentiation gradient objects that are
        necessary to evaluate the gradient of the ELBO.

        Returns:
            None
        """
        self.grad_reparameterization_variational_params = grad(
            variational_inference_utils.conduct_reparameterization
        )
        self.grad_log_priors = [grad(log_prior.logpdf) for log_prior in self.prior_obj_list]
        self.grad_log_variational_distr_params = (
            self.variational_distribution_obj.grad_logpdf_sample
        )

    def calculate_grad_reparameterization_variational_params(self, variational_params, sample):
        """Calculate the gradient of the reparameterization w.r.t. the varional
        parameters, evaluated at the given sample location.

        Args:
            variational_params (np.array): Array containing the variational parameters
            sample (np.array): Sample of the variational distribution

        Returns:
            grad_reparameterization (np.array): Value of the gradient of the reparameterization
                                                w.r.t. the variational parameters
        """
        grad_reparameterization = self.grad_reparameterization_variational_params(
            variational_params, sample
        )
        return grad_reparameterization

    def calculate_grad_log_prior_params(self, params):
        """Calculate the gradient of the log-prior distribution w.r.t. the
        random parameters.

        Args:
            params (np.array): Current parameter samples

        Returns:
            grad_variational (np.array): Gradient of log-prior distribution w.r.t. the
                                         random parameters,
                                         evaluated at the parameter value
        """
        grad_log_prior_lst = []
        for (dim, grad_log_prior_distr), param in zip(enumerate(self.grad_log_priors), params):
            grad_log_prior_lst.append(grad_log_prior_distr(param))

        grad_log_priors = np.array(grad_log_prior_lst)
        return grad_log_priors

    def calculate_grad_log_likelihood_params(self, params):
        """Calculate the gradient/jacobian of the log-likelihood function
        w.r.t. the random parameters, evaluated at the samples of the latter.

        Args:
            params (np.array): Current sample values of the random parameters


        Returns:
            jacobi_log_likelihood (np.array): Jacobian of the log-likelihood function
            log_likelihood (float): Value of the log-likelihood function
        """
        # TODO: For now this is done via finite differences;
        # parsing of adjoint gradient should be enabled too
        grad_log_likelihood = []

        log_likelihood = self.eval_log_likelihood(params)

        # two-point finite difference scheme
        for num, param in enumerate(params):
            zero_vec = np.zeros(params.shape)
            zero_vec[num] = self.finite_difference_step
            log_likelihood_right = self.eval_log_likelihood(params + zero_vec)

            grad_log_likelihood.append(
                (log_likelihood_right - log_likelihood) / self.finite_difference_step
            )

        jacobi_log_likelihood = np.array(grad_log_likelihood)

        return jacobi_log_likelihood, log_likelihood

    def _calculate_elbo(self, log_unnormalized_posterior_mean):
        """Calculate the ELBO of the current variational approximation.

        Args:
            log_unnormalized_posterior_mean (float): Monte-Carlo expectation of the
                                                     log-unnormalized posterior

        Returns:
            None
        """
        mu = np.array(self.variational_params[: self.num_variables])
        cov = np.diag(
            np.exp(
                2 * np.array(self.variational_params[self.num_variables : 2 * self.num_variables])
            ).flatten()
        )
        elbo = mvn.entropy(mu.flatten(), cov) + log_unnormalized_posterior_mean
        self.elbo_lst.append(elbo.flatten())

        # clear internal variables
        self.log_variational_mat = None
        self.log_posterior_unnormalized = None
        self.grad_log_variational_mat = None

    def _verbose_output(self):
        """Give some informative outputs during the BBVI iterations.

        Returns:
            None
        """
        if self.iteration_num % 10 == 0:
            mean_change = np.mean(np.abs(self.relative_change_variational_params))

            _logger.info("------------------------------------------------------------------------")
            _logger.info(f"Iteration {self.iteration_num + 1} of VI algorithm")
            _logger.info(f"So far {self.n_sims} simulation runs")
            _logger.info(f"The elbo is: {self.elbo_lst[-1]}")
            _logger.info(
                f"Mean absolute percentage change of all variational parameters: "
                f""
                f"{mean_change:.3f} %"
            )
            _logger.info(f"Values of variational parameters: \n")
            pprint.pprint(self.variational_params)
            _logger.info("------------------------------------------------------------------------")

    def stochastic_ascent_adam(self, gradient_estimate_x, x_vec, b1=0.9, b2=0.999, eps=10 ** -8):
        """Stochastic gradient ascent algorithm ADAM. Adam as described in
        http://arxiv.org/pdf/1412.6980.pdf. It's basically RMSprop with
        momentum and some correction terms.

        Args:
            gradient_estimate_x (np.array): (Noisy) Monte-Carlo estimate of a gradient
            x_vec (np.array): Current position in the latent design space
            b1 (float): The exponential decay rate for the first moment estimates
            b2 (float): The exponential decay rate for the second-moment estimates
            eps (float): Is a very small number to prevent any division by zero in the
                         implementation

        Returns:
            x_vec_new (lst): Updated vector/point in the latent design space
        """
        x_vec = x_vec.reshape(-1, 1)
        g = gradient_estimate_x
        self.m_param_adams = (1 - b1) * g + b1 * self.m_param_adams  # First moment estimate.
        self.v_param_adams = (1 - b2) * (
            g ** 2
        ) + b2 * self.v_param_adams  # Second moment estimate.
        mhat = self.m_param_adams / (1 - b1 ** (self.iteration_num + 1))  # Bias correction.
        vhat = self.v_param_adams / (1 - b2 ** (self.iteration_num + 1))
        x_vec_new = x_vec + self.learning_rate * mhat / (np.sqrt(vhat) + eps)  # update
        # TODO we should change the data type to np.array
        return x_vec_new.flatten()

    def _get_FIM(self):
        """Calculate the Fisher information matrix of the current variational
        approximation.

        Returns:
            fisher (np.array): fisher information matrix of the variational distribution
        """
        # TODO generalize this function for non Gaussians using MC
        # TODO generalize damping
        # Damping for the FIM in order to avoid a slowdown in the first few iterations (is
        # useful when the variance of the variational distribution is already small at the start
        # of the optimization)
        damping_lower_bound = 1e-2
        if self.iteration_num > 50:
            damping_coefficient = damping_lower_bound * np.exp(-(self.iteration_num - 50) / 50)
        else:
            damping_coefficient = damping_lower_bound
        fisher_diag = (
            np.exp(-2 * self.variational_params[self.num_variables :]) + damping_coefficient
        )
        fisher_diag = np.append(fisher_diag.flatten(), 2 * np.ones(self.num_variables))
        fisher = np.diag(fisher_diag)
        return fisher

    def _update_variational_params(self, elbo_gradient):
        """Update the variational parameters of the variational distribution
        based on learning rate rho_to and the noisy ELBO gradients.

        Args:
            elbo_gradient (np.array): Gradient vector of the ELBO w.r.t. the variational parameters

        Returns:
            None
        """
        self.variational_params_array = np.hstack(
            (self.variational_params_array, self.variational_params.reshape(-1, 1))
        )

        old_variational_params = self.variational_params.reshape(-1, 1)

        if self.natural_gradient_bool:
            FIM = self._get_fim()
            elbo_gradient = np.linalg.solve(FIM, elbo_gradient)

        if self.clipping_bool:
            # Clipping, in order to avoid exploding gradients
            # TODO move this output to the input?
            gradient_norm = (np.sum(elbo_gradient ** 2)) ** 0.5
            if gradient_norm > self.gradient_clipping_norm_threshold:
                _logger.info("Clipping gradient")
                elbo_gradient = (
                    elbo_gradient / gradient_norm * self.gradient_clipping_norm_threshold
                )

        # Use Adam for stochastic optimization
        self.variational_params = self.stochastic_ascent_adam(
            elbo_gradient, self.variational_params.reshape(-1, 1)
        )

        self._get_percentage_change_params(old_variational_params)

    def _get_percentage_change_params(self, old_variational_params):
        """Calculate L2 norm of the percentage change of the variational
        parameters.

        Args:
            old_variational_params (np.array): Array of variational parameters

        Returns:
            None
        """
        rel_distance_vec = np.divide(
            (self.variational_params.flatten() - old_variational_params.flatten()),
            old_variational_params.flatten(),
        )
        if len(self.relative_change_variational_params) > 0:
            self.relative_change_variational_params.append(rel_distance_vec)
        else:
            self.relative_change_variational_params = [rel_distance_vec]

        if np.any(np.isnan(self.relative_change_variational_params[-1])):
            self.relative_change_variational_params.append(1)  # dummy value to redo iteration

    def _initialize_variational_params(self):
        """
        Initialize the variational parameters. There are two possibilities:
            1. Random initialization:
                Is handeled by the variational distribution object
            2. Initialization based on the prior modeling (only for normal distributions!)
                Extract the prior moments and initialize the parameters based on them

        Returns:
            None

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
                    f"Initializing the variational parameters based on the prior is only possible "
                    f"for distribution family 'normal'"
                )
        else:
            valid_initialization_types = {"random", "prior"}
            raise NotImplementedError(
                f"{self.variational_params_initialization_approach} is not known.\n"
                f"Valid options are {valid_initialization_types}"
            )

        self.variational_params_array = np.empty((len(self.variational_params), 0))

    def _initialize_variational_params_from_prior(self, random_variables):
        """Initializes the variational parameters based on the prior
        definition. The variational distribution might be transformed in a
        second step such that the actual variational distribution is of a
        different family. Only is used for normal distributions.

        Args:
            random_variables (dict): Dictionary containing the prior probabilistic description of
            the input variables

        Returns:
            None
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

    def post_run(self):
        """Write results and potentially visualize them using the visualization
        module.

        Returns:
            None
        """
        if self.result_description["write_results"]:
            result_dict = self._prepare_result_description()
            write_results(
                result_dict,
                self.global_settings["output_dir"],
                self.global_settings["experiment_name"],
            )
        vis.vi_visualization_instance.save_plots()

    def _prepare_result_description(self):
        """Transform the results back the correct space and summarize results
        in a dictionary that will be stored as a pickle file.

        Returns:
            result_description (dict): Dictionary with result summary of the analysis
        """
        result_description = {
            "final_elbo": self.elbo_lst[-1],
            "iterations": self.iteration_num,
            "batch_size": self.n_samples_per_iter,
            "number_of_sim": self.n_sims,
            "learning_rate": self.learning_rate,
            "variational_parameter_initialization": self.variational_params_initialization_approach,
        }

        if self.export_quantities_over_iter:
            result_description.update(
                {
                    "params_over_iter": self.variational_params_array,
                    "likelihood_noise_var": self.noise_lst,
                    "elbo": self.elbo_lst,
                }
            )

        distribution_dict = self.variational_distribution_obj.export_dict(self.variational_params)
        if self.variational_transformation == "exp":
            distribution_dict.update(
                {"variational_transformation": self.variational_transformation}
            )

        result_description.update({"variational_distr": distribution_dict})
        return result_description

    def eval_model(self):
        """Evaluate model for the sample batch.

        Returns:
           result_dict (dict): Dictionary containing model response for sample batch
        """
        result_dict = self.model.evaluate()
        return result_dict

    def _initialize_prior_model(self):
        """Initialize the prior model of the inverse problem form the problem
        description.

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
                'Variational inference for random fields is not yet implemented! Abort...'
            )
        # for each rv, evaluate prior all corresponding dims in current sample batch (column of
        # mat corresponds to all realization for this variable)
        for dim, rv_options in enumerate(random_variables.values()):
            if rv_options['distribution'] == 'normal' and rv_options['size'] == 1:
                rv_options['distribution_parameter'][1] = 1
            self.prior_obj_list.append(mcmc_utils.create_proposal_distribution(rv_options))

    def eval_log_likelihood(self, params):
        """Calculate the log-likelihood of the observation data. Evaluation of
        the likelihood model for all inputs of the sample batch will trigger
        the actual forward simulation (can be executed in parallel as batch-
        sequential procedure)

        Args:
            sample_batch (np.array): Sample-batch with samples row-wise

        Returns:
            log_likelihood (np.array): Vector of the log-likelihood function for all input
                                       samples of the current batch
        """
        # The first samples belong to simulation input
        # get simulation output (run actual forward problem)--> data is saved to DB

        self.model.update_model_from_sample_batch(np.atleast_2d(params).reshape(1, -1))
        log_likelihood = self.eval_model()
        self.noise_lst.append(self.model.noise_var)

        return log_likelihood

    def get_log_prior(self, params):
        """Construct and evaluate the log prior of the model for current sample
        batch.

        Args:
            sample_batch (np.array): Sample batch for which the model should be evaluated

        Returns:
            log_prior (np.array): log-prior vector evaluated for current sample batch
        """
        log_prior = 0
        for dim, prior_distr in enumerate(self.prior_obj_list):
            log_prior += prior_distr.logpdf(params[dim])

        return log_prior

    def _get_fim(self):
        """Get the FIM for the current variational distribution and add
        dampening if desired.

        Returns:
            fisher (np.array): fisher information matrix of the variational distribution
        """
        FIM = self.variational_distribution_obj.fisher_information_matrix(self.variational_params)
        if self.fim_dampening_bool:
            if self.iteration_num > self.fim_decay_start_iter:
                dampening_coefficient = self.fim_dampening_coefficient * np.exp(
                    -(self.iteration_num - self.fim_decay_start_iter) / self.fim_decay_start_iter
                )
                dampening_coefficient = max(self.fim_dampening_lower_bound, dampening_coefficient)
            else:
                dampening_coefficient = self.fim_dampening_coefficient
            FIM = FIM + np.eye(len(FIM)) * dampening_coefficient
        return FIM
