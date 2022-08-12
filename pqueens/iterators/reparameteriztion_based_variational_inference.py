"""Reparameterization trick based variational inference."""
import logging

import numpy as np
from scipy.stats import multivariate_normal as mvn

from pqueens.iterators.variational_inference import (
    VALID_EXPORT_FIELDS,
    VariationalInferenceIterator,
)
from pqueens.utils import variational_inference_utils
from pqueens.utils.collection_utils import CollectionObject
from pqueens.utils.fd_jacobian import fd_jacobian, get_positions
from pqueens.utils.valid_options_utils import check_if_valid_option, get_option, get_valid_options

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

    The current implementation does not support importance sampling of the MC gradient.

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
        score_function_bool (bool): Boolean flag to decide wheater the score function term
                                    should be considered in the elbo gradient. If true the
                                    score function is considered.
        finite_difference_step (float): Finite difference step size
        finite_difference_method (str): Method to calculate a finite difference based
                                        approximation of the Jacobian matrix:

                        - '2-point': a one sided scheme by definition
                        - '3-point': more exact but needs twice as many function evaluations
        likelihood_gradient_method (str): Method for how to calculate the gradient
                                                        of the log-likelihood

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
        num_parameters,
        natural_gradient_bool,
        fim_dampening_bool,
        fim_decay_start_iter,
        fim_dampening_coefficient,
        fim_dampening_lower_bound,
        variational_distribution_obj,
        stochastic_optimizer,
        score_function_bool,
        finite_difference_step,
        likelihood_gradient_method,
        finite_difference_method,
        iteration_data,
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
            num_parameters (int): Actual number of model input parameters that should be calibrated
            natural_gradient_bool (boolean): True if natural gradient should be used
            fim_dampening_bool (boolean): True if FIM dampening should be used
            fim_decay_start_iter (float): Iteration at which the FIM dampening is started
            fim_dampening_coefficient (float): Initial nugget term value for the FIM dampening
            fim_dampening_lower_bound (float): Lower bound on the FIM dampening coefficient
            variational_distribution_obj (VariationalDistribution): Variational distribution object
            stochastic_optimizer (obj): QUEENS stochastic optimizer object
            score_function_bool (bool): Boolean flag to decide whether the score function term
                                    should be considered in the ELBO gradient. If true the
                                    score function is considered.
            finite_difference_step (float): Finite difference step size
            likelihood_gradient_method (str, optional): Method for how to calculate the gradient
                                                        of the log-likelihood
            finite_difference_method (str): Method to calculate a finite difference based
                                            approximation of the Jacobian matrix:

                            - '2-point': a one sided scheme by definition
                            - '3-point': more exact but needs twice as many function evaluations
            iteration_data (CollectionObject): Object to store iteration data if desired
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
            num_parameters,
            natural_gradient_bool,
            fim_dampening_bool,
            fim_decay_start_iter,
            fim_dampening_coefficient,
            fim_dampening_lower_bound,
            variational_distribution_obj,
            stochastic_optimizer,
            iteration_data,
        )
        self.score_function_bool = score_function_bool
        self.finite_difference_step = finite_difference_step
        self.likelihood_gradient_method = likelihood_gradient_method
        self.finite_difference_method = finite_difference_method

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
        valid_finite_difference_methods = ["2-point", "3-point"]
        finite_difference_method = method_options.get("finite_difference_method")
        if likelihood_gradient_method == "finite_difference":
            check_if_valid_option(valid_finite_difference_methods, finite_difference_method)

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
            num_parameters,
            natural_gradient_bool,
            fim_dampening_bool,
            fim_decay_start_iter,
            fim_dampening_coefficient,
            fim_dampening_lower_bound,
            variational_distribution_obj,
            stochastic_optimizer,
        ) = super().get_base_attributes_from_config(config, iterator_name)

        iterative_data_names = get_valid_options(
            VALID_EXPORT_FIELDS,
            method_options["result_description"].get("iterative_field_names", []),
        )
        iteration_data = CollectionObject(*iterative_data_names)
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
            num_parameters=num_parameters,
            natural_gradient_bool=natural_gradient_bool,
            fim_dampening_bool=fim_dampening_bool,
            fim_decay_start_iter=fim_decay_start_iter,
            fim_dampening_coefficient=fim_dampening_coefficient,
            fim_dampening_lower_bound=fim_dampening_lower_bound,
            variational_distribution_obj=variational_distribution_obj,
            stochastic_optimizer=stochastic_optimizer,
            score_function_bool=score_function_bool,
            finite_difference_step=finite_difference_step,
            likelihood_gradient_method=likelihood_gradient_method,
            finite_difference_method=finite_difference_method,
            iteration_data=iteration_data,
        )

    def core_run(self):
        """Core run for variational inference with reparameterization trick."""
        _logger.info('Starting reprarameterization based variational inference...')
        super().core_run()

    def _calculate_elbo_gradient(self, variational_parameters):
        """Compute ELBO gradient with reparameterization trick.

         Some clarification of terms:
         sample_mat: sample of the variational distribution
         variational_params: variational parameters that parameterize var. distr. of params
                             (lambda)

        Args:
            variational_parameters (np.ndarray): variational parameters

        Returns:
            np.ndarray: ELBO gradient (n_params x 1)
        """
        # update variational_params
        self.variational_params = variational_parameters.flatten()

        (
            sample_batch,
            standard_normal_sample_batch,
        ) = self.variational_distribution_obj.conduct_reparameterization(
            self.variational_params, self.n_samples_per_iter
        )
        jacobi_reparameterization_lst = (
            self.variational_distribution_obj.jacobi_variational_params_reparameterization(
                standard_normal_sample_batch, self.variational_params
            )
        )

        grad_log_prior_lst = self.calculate_grad_log_prior_params(sample_batch)
        log_prior_lst = self.get_log_prior(sample_batch)
        grad_variational_lst = self.variational_distribution_obj.grad_logpdf_sample(
            sample_batch, self.variational_params
        )

        # check if score function should be added to the derivative
        if self.score_function_bool:
            # pylint: disable=line-too-long
            grad_log_variational_distr_variational_params_lst = (
                variational_inference_utils.calculate_grad_log_variational_distr_variational_params(
                    jacobi_reparameterization_lst, grad_variational_lst
                )
            )
            # pylint: enable=line-too-long

        log_likelihood_batch, grad_log_likelihood_batch = self.calculate_grad_log_likelihood_params(
            sample_batch
        )
        sample_elbo_grad = np.zeros((1, self.variational_params.size))
        log_unnormalized_posterior = 0
        for (
            grad_log_likelihood,
            grad_log_prior,
            grad_variational,
            jacobi_reparameterization,
            grad_log_variational_distr_variational_params,
            log_likelihood,
            log_prior,
        ) in zip(
            grad_log_likelihood_batch,
            grad_log_prior_lst,
            grad_variational_lst,
            jacobi_reparameterization_lst,
            grad_log_variational_distr_variational_params_lst,
            log_likelihood_batch,
            log_prior_lst,
        ):

            # calculate the elbo gradient for one sample
            sample_elbo_grad += np.dot(
                (grad_log_likelihood + grad_log_prior - grad_variational).T,
                jacobi_reparameterization.T,
            )
            if self.score_function_bool:
                sample_elbo_grad = (
                    sample_elbo_grad - grad_log_variational_distr_variational_params.T
                )

            # calculate the unnormalized posterior for one sample
            log_unnormalized_posterior += log_likelihood + np.sum(log_prior)

        # MC estimate of elbo gradient
        grad_elbo = sample_elbo_grad / self.n_samples_per_iter
        # MC estimate for unnormalized posterior
        log_unnormalized_posterior = log_unnormalized_posterior / self.n_samples_per_iter

        self._calculate_elbo(log_unnormalized_posterior)
        return grad_elbo.reshape(-1, 1)

    def calculate_grad_log_prior_params(self, sample_batch):
        """Gradient of the log-prior distribution w.r.t. the random parameters.

        Args:
            sample_batch (np.array): Current parameter sample batch

        Returns:
            grad_log_prior_lst (lst): List of gradients of log-prior distribution w.r.t. the
                                      input parameters,
                                      evaluated at the parameter value
        """
        grad_log_prior_lst = []
        for parameter, sample in zip(self.parameters.to_list(), sample_batch):
            grad_log_prior_lst.append(parameter.distribution.grad_logpdf(sample).reshape(-1))

        grad_log_priors = np.array(grad_log_prior_lst).flatten()
        return grad_log_priors

    def calculate_grad_log_likelihood_params(self, sample_batch):
        """Calculate the gradient/jacobian of the log-likelihood function.

        Gradient is calculated w.r.t. the argument params not the variational parameters.

        Args:
            sample_batch (np.array): Current sample_batch of the random parameters

        Returns:
            jacobi_log_likelihood (np.array): Jacobian of the log-likelihood function
            log_likelihood (float): Value of the log-likelihood function
        """
        gradient_methods = {
            'provided_gradient': self._calculate_grad_log_lik_params_provided_gradient,
            'finite_difference': self._calculate_grad_log_lik_params_finite_difference,
        }
        my_gradient_method = get_option(gradient_methods, self.likelihood_gradient_method)
        log_likelihood, grad_log_likelihood = my_gradient_method(sample_batch)
        return log_likelihood, grad_log_likelihood

    def _calculate_grad_log_lik_params_provided_gradient(self, sample_batch):
        """Calculate gradient of log likelihood based on provided gradient.

        The gradient of the log-likelihood is calculated w.r.t. the input of the underlying forward
        model, here denoted by params.

        Args:
            sample_batch (np.array): Current sample batch of the random parameters

        Returns:
            grad_log_likelihood_lst (list): List of gradients of the log-likelihood function
                                            w.r.t. the input sample
            log_likelihood (float): Value of the log-likelihood function
        """
        log_likelihood, grad_log_likelihood_lst = self.eval_log_likelihood(
            sample_batch, gradient_bool=True
        )
        return log_likelihood, grad_log_likelihood_lst

    def _calculate_grad_log_lik_params_finite_difference(self, sample_batch):
        """Gradient of the log likelihood with finite differences.

        The gradient of the log-likelihood is calculated w.r.t. the input of the underlying forward
        model, here denoted by params.

        Args:
            sample_batch (np.array): Current sample batch of the random parameters

        Returns:
            grad_log_likelihood_lst (list): List with gradients of the log-likelihood function
                                            per sample, w.r.t. to the current sample
            log_likelihood_batch (float): Value of the log-likelihood function for the batch of
                                          input samples
        """
        sample_stencil_lst = []
        delta_positions_lst = []
        for sample in sample_batch:
            sample_stencil, delta_positions = get_positions(
                sample,
                method=self.finite_difference_method,
                rel_step=self.finite_difference_step,
                bounds=[-np.inf, np.inf],
            )
            sample_stencil_lst.append(sample_stencil)
            delta_positions_lst.append(delta_positions)

        sample_stencil_batch = np.array(sample_stencil_lst).reshape(
            -1, sample_stencil_lst[0].shape[1]
        )

        # model response should now correspond to objective function evaluated at positions
        all_log_likelihood_batch = self.eval_log_likelihood(sample_stencil_batch)

        # get actual likelihood and stencil points
        num_jump = int(all_log_likelihood_batch.size / self.n_samples_per_iter)
        log_likelihood_batch = all_log_likelihood_batch[0:-1:num_jump]
        perturbed_log_likelihood_batch = np.delete(
            all_log_likelihood_batch, np.arange(0, all_log_likelihood_batch.size, num_jump)
        ).reshape(self.n_samples_per_iter, -1)

        grad_log_likelihood_lst = []
        for log_likelihood, delta_positions, perturbed_log_likelihood in zip(
            log_likelihood_batch, delta_positions_lst, perturbed_log_likelihood_batch
        ):
            grad_log_likelihood_lst.append(
                fd_jacobian(
                    log_likelihood.reshape(1, 1),
                    perturbed_log_likelihood.reshape(1, -1),
                    delta_positions.reshape(1, -1),
                    False,
                    method=self.finite_difference_method,
                )
            )

        return log_likelihood_batch, grad_log_likelihood_lst

    def _calculate_elbo(self, log_unnormalized_posterior_mean):
        """Calculate the ELBO of the current variational approximation.

        Args:
            log_unnormalized_posterior_mean (float): Monte-Carlo expectation of the
                                                     log-unnormalized posterior

        Returns:
            None
        """
        mean = np.array(self.variational_params[: self.num_parameters])
        covariance = np.diag(
            np.exp(
                2 * np.array(self.variational_params[self.num_parameters : 2 * self.num_parameters])
            ).flatten()
        )
        elbo = float(mvn.entropy(mean.flatten(), covariance) + log_unnormalized_posterior_mean)
        self.iteration_data.add(elbo=elbo)

    def _verbose_output(self):
        """Give some informative outputs during the VI iterations."""
        _logger.info("-" * 80)
        _logger.info(f"Iteration {self.stochastic_optimizer.iteration + 1} of RPVI algorithm")

        super()._verbose_output()

        if self.stochastic_optimizer.iteration > 1:
            _logger.info(f"Likelihood noise variance: {self.model.normal_distribution.covariance}%")
        _logger.info("-" * 80)

    def _prepare_result_description(self):
        """Creates the dictionary for the result pickle file.

        Returns:
            result_description (dict): Dictionary with result summary of the analysis
        """
        result_description = super()._prepare_result_description()
        if self.iteration_data:
            result_description["iteration_data"].update(self.iteration_data.to_dict())
        return result_description

    def eval_log_likelihood(self, sample_batch, gradient_bool=False):
        """Calculate the log-likelihood of the observation data.

        Evaluation of the likelihood model for all inputs of the sample batch will trigger
        the actual forward simulation (can be executed in parallel as batch-
        sequential procedure)

        Args:
            sample_batch (np.array): Sample-batch with samples row-wise
            gradient_bool (bool): Flag to determine, whether gradient should be provided as well.

        Returns:
            likelihood_return_tuple (tuple): Tuple containing Vector of the log-likelihood
                                             function for all inputs samples of the current batch,
                                             and potentially the corresponding gradient
        """
        # The first samples belong to simulation input
        # get simulation output (run actual forward problem)--> data is saved to DB
        likelihood_return_tuple = self.model.evaluate(
            sample_batch.reshape(-1, self.num_parameters), gradient_bool=gradient_bool
        )
        self.n_sims += len(sample_batch)
        self.iteration_data.add(
            n_sims=self.n_sims,
            likelihood_variance=self.model.normal_distribution.covariance,
            samples=sample_batch,
        )

        return likelihood_return_tuple

    def get_log_prior(self, sample_batch):
        """Evaluate the log prior of the model for current sample batch.

        Args:
            sample_batch (np.array): Sample batch for which the model should be evaluated

        Returns:
            log_prior (np.array): log-prior vector evaluated for current sample batch
        """
        log_prior = self.parameters.joint_logpdf(sample_batch)
        return log_prior
