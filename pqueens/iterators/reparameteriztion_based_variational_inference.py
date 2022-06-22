"""Reparameterization trick based variational inference."""
import logging

import numpy as np
from scipy.stats import multivariate_normal as mvn

from pqueens.iterators.variational_inference import VariationalInferenceIterator
from pqueens.utils import variational_inference_utils
from pqueens.utils.valid_options_utils import get_option

_logger = logging.getLogger(__name__)


class RPVIIterator(VariationalInferenceIterator):
    """Reparameterization based variational inference (RPVI).

    Iterator for Bayesian inverse problems. This variational inference approach requires
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
           global_settings (dict): Global settings of the QUEENS simulations
        model (obj): Underlying simulation model on which the inverse analysis is conducted
        result_description (dict): Settings for storing and visualizing the results
        db (obj): QUEENS database object
        experiment_name (str): Name of the QUEENS simulation
        variational_family (str): Density type for variational approximation
        variational_params_initialization_approach (str): Flag to decide how to initialize the
                                                          variational parameters
        n_samples_per_iter (int): Batch size per iteration (number of simulations per iteration to
                                  estimate the involved expectations)
        variational_transformation (str): String encoding the transformation that will be applied to
                                          the variational density
        random_seed (int): Seed for the random number generators
        max_feval (int): Maximum number of simulation runs for this analysis
        num_variables (int): Actual number of model input variables that should be calibrated
        natural_gradient_bool (boolean): True if natural gradient should be used
        fim_dampening_bool (boolean): True if FIM dampening should be used
        fim_decay_start_iter (float): Iteration at which the FIM dampening is started
        fim_dampening_coefficient (float): Initial nugget term value for the FIM dampening
        fim_dampening_lower_bound (float): Lower bound on the FIM dampening coefficient
        export_quantities_over_iter (boolean): True if data (variational_params, elbo) should
                                               be exported in the pickle file
        n_sims (int): Number of probabilistic model calls
        variational_distribution_obj (VariationalDistribution): Variational distribution object
        variational_params (np.array): Row-vector containing the variational parameters
        elbo_list (list): ELBO value of every iteration
        prior_obj_list (list): List containing objects of prior models for the random input. The
                               list is ordered in the same way as the random input definition in
                               the input file
        parameter_list (list): List of parameters from previous iterations for the ISMC gradient
        variational_params_list (list): List of parameters from first to last iteration
        model_eval_iteration_period (int): If the iteration number is a multiple of this number
                                           the probabilistic model is sampled independent of the
                                           other conditions
        stochastic_optimizer (obj): QUEENS stochastic optimizer object
        score_function_bool (bool): Boolean flag to decide whether the score function term
                                    should be considered in the ELBO gradient. If true the
                                    score function is considered.
        finite_difference_step (float): Finite difference step size

    Returns:
        rpvi_obj (obj): Instance of the RPVIIterator
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
        natural_gradient_bool,
        fim_dampening_bool,
        fim_decay_start_iter,
        fim_dampening_coefficient,
        fim_dampening_lower_bound,
        export_quantities_over_iter,
        variational_distribution_obj,
        stochastic_optimizer,
        score_function_bool,
        finite_difference_step,
        likelihood_gradient_method,
    ):
        """Initialize RPVI iterator.

        Args:
            global_settings (dict): Global settings of the QUEENS simulations
            model (obj): Underlying simulation model on which the inverse analysis is conducted
            result_description (dict): Settings for storing and visualizing the results
            db (obj): QUEENS database object
            experiment_name (str): Name of the QUEENS simulation
            variational_family (str): Density type for variational approximation
            variational_params_initialization_approach (str): Flag to decide how to initialize the
                                                              variational parameters
            n_samples_per_iter (int): Batch size per iteration (number of simulations per iteration
                                                to estimate the involved expectations)
            variational_transformation (str): String encoding the transformation that will be
                                              applied to
                                              the variational density
            random_seed (int): Seed for the random number generators
            max_feval (int): Maximum number of simulation runs for this analysis
            num_variables (int): Actual number of model input variables that should be calibrated
            natural_gradient_bool (boolean): True if natural gradient should be used
            fim_dampening_bool (boolean): True if FIM dampening should be used
            fim_decay_start_iter (float): Iteration at which the FIM dampening is started
            fim_dampening_coefficient (float): Initial nugget term value for the FIM dampening
            fim_dampening_lower_bound (float): Lower bound on the FIM dampening coefficient
            export_quantities_over_iter (boolean): True if data (variational_params, elbo, ESS)
                                                   should be exported in the pickle file
            variational_distribution_obj (VariationalDistribution): Variational distribution object
            stochastic_optimizer (obj): QUEENS stochastic optimizer object
            score_function_bool (bool): Boolean flag to decide whether the score function term
                                    should be considered in the ELBO gradient. If true the
                                    score function is considered.
            finite_difference_step (float): Finite difference step size
            likelihood_gradient_method (str, optional): Method for how to calculate the gradient
                                                        of the log-likelihood
        """
        super().__init__(
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
            natural_gradient_bool,
            fim_dampening_bool,
            fim_decay_start_iter,
            fim_dampening_coefficient,
            fim_dampening_lower_bound,
            export_quantities_over_iter,
            variational_distribution_obj,
            stochastic_optimizer,
        )
        self.score_function_bool = score_function_bool
        self.finite_difference_step = finite_difference_step
        self.likelihood_gradient_method = likelihood_gradient_method
        self.noise_list = []

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create RPVI iterator from problem description.

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator (optional)
            model (model): Model to use (optional)

        Returns:
            iterator (obj): RPVI object
        """
        method_options = config[iterator_name]['method_options']
        score_function_bool = method_options.get("score_function_bool", True)
        finite_difference_step = method_options.get("finite_difference_step")
        likelihood_gradient_method = method_options.get(
            "likelihood_gradient_method", "finite_difference"
        )

        (
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
            natural_gradient_bool,
            fim_dampening_bool,
            fim_decay_start_iter,
            fim_dampening_coefficient,
            fim_dampening_lower_bound,
            variational_distribution_obj,
            stochastic_optimizer,
            export_quantities_over_iter,
        ) = super().get_base_attributes_from_config(config, iterator_name)

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
            natural_gradient_bool=natural_gradient_bool,
            fim_dampening_bool=fim_dampening_bool,
            fim_decay_start_iter=fim_decay_start_iter,
            fim_dampening_coefficient=fim_dampening_coefficient,
            fim_dampening_lower_bound=fim_dampening_lower_bound,
            export_quantities_over_iter=export_quantities_over_iter,
            variational_distribution_obj=variational_distribution_obj,
            stochastic_optimizer=stochastic_optimizer,
            score_function_bool=score_function_bool,
            finite_difference_step=finite_difference_step,
            likelihood_gradient_method=likelihood_gradient_method,
        )

    def core_run(self):
        """Core run for variational inference with reparameterization trick."""
        _logger.info('Starting reprarameterization based variational inference...')
        super().core_run()

    def _calculate_elbo_gradient(self, variational_parameters):

        # Some clarification of terms:
        # params: actual parameters for which we solve the inverse problem
        # sample: sample of the variational distribution
        # variational_params: variational parameters that parameterize var,.distr. of params
        #                     (lambda)
        variational_params = variational_parameters.flatten()

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
            variational_params = self.variational_params
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
                    variational_inference_utils.grad_varparams_reparameterization(
                        var_params_sample_dim, sample_dim
                    )
                )

            # evaluate the necessary contributions
            params = np.array(params)
            grad_reparameterization = np.array(grad_reparameterization).reshape(-1, 1)

            grad_log_prior = self.calculate_grad_log_prior_params(params)
            log_prior = self.get_log_prior(params)
            grad_variational = (
                variational_inference_utils.calculate_grad_log_variational_distr_params(
                    self.variational_distribution_obj.grad_logpdf_sample, params, variational_params
                )
            )

            # check if score function should be added to the derivative
            if self.score_function_bool:
                # pylint: disable=line-too-long
                grad_log_variational_distr_variational_params = variational_inference_utils.calculate_grad_log_variational_distr_variational_params(
                    grad_reparameterization, grad_variational
                )
                # pylint: enable=line-too-long

            log_likelihood, grad_log_likelihood = self.calculate_grad_log_likelihood_params(params)

            # calculate the elbo gradient for one sample
            sample_elbo_grad += (
                np.vstack((grad_log_likelihood.reshape(-1, 1), grad_log_likelihood.reshape(-1, 1)))
                + np.vstack((grad_log_prior.reshape(-1, 1), grad_log_prior.reshape(-1, 1)))
                - np.vstack((grad_variational.reshape(-1, 1), grad_variational.reshape(-1, 1)))
            ) * grad_reparameterization.reshape(-1, 1)
            if self.score_function_bool:
                sample_elbo_grad = sample_elbo_grad - grad_log_variational_distr_variational_params

            # calculate the unnormalized posterior for one sample
            log_unnormalized_posterior += log_likelihood + log_prior

        # MC estimate of elbo gradient
        grad_elbo = sample_elbo_grad / self.n_samples_per_iter
        # MC estimate for unnormalized posterior
        log_unnormalized_posterior = log_unnormalized_posterior / self.n_samples_per_iter

        self._calculate_elbo(log_unnormalized_posterior)
        return grad_elbo.reshape(-1, 1)

    def calculate_grad_log_prior_params(self, params):
        """Gradient of the log-prior distribution w.r.t. the random parameters.

        Args:
            params (np.array): Current parameter samples

        Returns:
            grad_variational (np.array): Gradient of log-prior distribution w.r.t. the
                                         random parameters,
                                         evaluated at the parameter value
        """
        grad_log_prior_lst = []
        for prior_distr, param in zip(self.prior_obj_list, params):
            grad_log_prior_lst.append(prior_distr.grad_logpdf(param))

        grad_log_priors = np.array(grad_log_prior_lst)
        return grad_log_priors

    def calculate_grad_log_likelihood_params(self, params):
        """Calculate the gradient/jacobian of the log-likelihood function.

        Gradient is calculated w.r.t. the argument params

        Args:
            params (np.array): Current sample values of the random parameters

        Returns:
            jacobi_log_likelihood (np.array): Jacobian of the log-likelihood function
            log_likelihood (float): Value of the log-likelihood function
        """
        gradient_methods = {
            'adjoint': self._calculate_grad_log_lik_params_adjoint,
            'finite_difference': self._calculate_grad_log_lik_params_finite_difference,
        }
        my_gradient_method = get_option(gradient_methods, self.likelihood_gradient_method)
        log_likelihood, jacobi_log_likelihood = my_gradient_method(params)
        return log_likelihood, jacobi_log_likelihood

    def _calculate_grad_log_lik_params_adjoint(self, params):
        """Calculate the gradient of the log likelihood based on adjoints."""
        log_likelihood, grad_log_likelihood = self.eval_log_likelihood(params, gradient=True)
        return log_likelihood, grad_log_likelihood

    def _calculate_grad_log_lik_params_finite_difference(self, params):
        """Gradient of the log likelihood with finite differences."""
        grad_log_likelihood = []

        log_likelihood = self.eval_log_likelihood(params)

        # two-point finite difference scheme
        for num in np.arange(params.size):
            zero_vec = np.zeros(params.shape)
            zero_vec[num] = self.finite_difference_step
            log_likelihood_right = self.eval_log_likelihood(params + zero_vec)

            grad_log_likelihood.append(
                (log_likelihood_right - log_likelihood) / self.finite_difference_step
            )

        grad_log_likelihood = np.array(grad_log_likelihood)

        return log_likelihood, grad_log_likelihood

    def _calculate_elbo(self, log_unnormalized_posterior_mean):
        """Calculate the ELBO of the current variational approximation.

        Args:
            log_unnormalized_posterior_mean (float): Monte-Carlo expectation of the
                                                     log-unnormalized posterior

        Returns:
            None
        """
        mean = np.array(self.variational_params[: self.num_variables])
        covariance = np.diag(
            np.exp(
                2 * np.array(self.variational_params[self.num_variables : 2 * self.num_variables])
            ).flatten()
        )
        elbo = float(mvn.entropy(mean.flatten(), covariance) + log_unnormalized_posterior_mean)
        self.elbo_list.append(elbo)

    def _verbose_output(self):
        """Give some informative outputs during the VI iterations."""
        _logger.info("-" * 80)
        _logger.info(f"Iteration {self.stochastic_optimizer.iteration + 1} of RPVI algorithm")

        super()._verbose_output()

        if self.stochastic_optimizer.iteration > 1:
            rel_noise = (
                np.mean(np.abs(self.noise_list[-2] - self.noise_list[-1]) / self.noise_list[-2])
                * 100
            )
            _logger.info(
                f"Likelihood noise variance: {self.noise_list[-1]} (mean relative change "
                f"{rel_noise:.2f}) %"
            )
        _logger.info("-" * 80)

    def _prepare_result_description(self):
        """Creates the dictionary for the result pickle file.

        Returns:
            result_description (dict): Dictionary with result summary of the analysis
        """
        result_description = super()._prepare_result_description()
        if self.export_quantities_over_iter:
            result_description.update(
                {
                    "likelihood_noise_var": self.noise_list,
                }
            )
        return result_description

    def eval_model(self, gradient=False):
        """Evaluate model for the sample batch.

        Args:
            gradient (bool): Boolean to compute gradient of the model as well, if set to True

        Returns:
           result_dict (dict): Dictionary containing model response for sample batch
        """
        result_dict = self.model.evaluate(gradient=gradient)
        return result_dict

    def eval_log_likelihood(self, params, gradient=False):
        """Calculate the log-likelihood of the observation data.

        Evaluation of the likelihood model for all inputs of the sample batch will trigger
        the actual forward simulation (can be executed in parallel as batch-
        sequential procedure)

        Args:
            sample_batch (np.array): Sample-batch with samples row-wise
            gradient (bool): Flag to determine, whether gradient should be provided as well.

        Returns:
            likelihood_return_tuple (tuple): Tuple containing Vector of the log-likelihood
                                             function for all inputs samples of the current batch,
                                             and potentially the corresponding gradient
        """
        # The first samples belong to simulation input
        # get simulation output (run actual forward problem)--> data is saved to DB

        self.model.update_model_from_sample_batch(np.atleast_2d(params).reshape(1, -1))
        likelihood_return_tuple = self.eval_model(gradient=gradient)
        self.n_sims += len(params)
        self.n_sims_list.append(self.n_sims)
        self.noise_list.append(self.model.normal_distribution.covariance)

        return likelihood_return_tuple

    def get_log_prior(self, params):
        """Evaluate the log prior of the model for current sample batch.

        Args:
            sample_batch (np.array): Sample batch for which the model should be evaluated

        Returns:
            log_prior (np.array): log-prior vector evaluated for current sample batch
        """
        log_prior = 0
        for dim, prior_distr in enumerate(self.prior_obj_list):
            log_prior += prior_distr.logpdf(params[dim])

        return log_prior
