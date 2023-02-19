"""Reparameterization trick based variational inference."""
import logging

import numpy as np
from scipy.stats import multivariate_normal as mvn

from pqueens.iterators.variational_inference import (
    VALID_EXPORT_FIELDS,
    VariationalInferenceIterator,
)
from pqueens.utils.collection_utils import CollectionObject
from pqueens.utils.valid_options_utils import check_if_valid_options

_logger = logging.getLogger(__name__)


class RPVIIterator(VariationalInferenceIterator):
    """Reparameterization based variational inference (RPVI).

    Iterator for Bayesian inverse problems. This variational inference approach requires
    model gradients/Jacobians w.r.t. the parameters/the parameterization of the
    inverse problem. The latter can be provided by:

        - A finite differences approximation of the gradient/Jacobian, which requires in the
          simplest case d+1 additional solver calls
        - An externally provided gradient/Jacobian that was, e.g. calculated via adjoint methods or
          automated differentiation

    The current implementation does not support the importance sampling of the MC gradient.

    The mathematical details of the algorithm can be found in [1], [2], [3].

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
        score_function_bool (bool): Boolean flag to decide whether the score function term
                                    should be considered in the elbo gradient. If *True* the
                                    score function is considered.

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
        method_options = config[iterator_name]
        score_function_bool = method_options.get("score_function_bool", False)
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

        iterative_data_names = method_options["result_description"].get("iterative_field_names", [])
        check_if_valid_options(VALID_EXPORT_FIELDS, iterative_data_names)
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
            iteration_data=iteration_data,
        )

    def core_run(self):
        """Core run for variational inference with reparameterization trick."""
        _logger.info('Starting reparameterization based variational inference...')
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

        grad_log_priors = self.parameters.grad_joint_logpdf(sample_batch)
        log_priors = self.parameters.joint_logpdf(sample_batch)
        grad_variational_lst = self.variational_distribution_obj.grad_logpdf_sample(
            sample_batch, self.variational_params
        ).reshape(self.n_samples_per_iter, -1)

        log_likelihood_batch, grad_log_likelihood_batch = self.evaluate_and_gradient(sample_batch)

        # calculate the elbo gradient per sample
        sample_elbo_grad = np.sum(
            (grad_log_likelihood_batch + grad_log_priors - grad_variational_lst)[:, np.newaxis, :]
            * jacobi_reparameterization_lst,
            axis=2,
        )

        # check if score function should be added to the derivative
        # Using the score function might lead to a larger variance in the estimate, in doubt turn
        # it off
        if self.score_function_bool:
            score_function = self.variational_distribution_obj.grad_params_logpdf(
                self.variational_params, sample_batch
            ).T
            sample_elbo_grad = sample_elbo_grad - score_function

        # MC estimate of elbo gradient
        grad_elbo = np.sum(sample_elbo_grad, axis=0) / self.n_samples_per_iter
        # MC estimate for unnormalized posterior
        log_unnormalized_posterior = (
            np.sum(log_likelihood_batch + log_priors) / self.n_samples_per_iter
        )

        self._calculate_elbo(log_unnormalized_posterior)
        return grad_elbo.reshape(-1, 1)

    def _calculate_elbo(self, log_unnormalized_posterior_mean):
        """Calculate the ELBO of the current variational approximation.

        Args:
            log_unnormalized_posterior_mean (float): Monte-Carlo expectation of the
                                                     log-unnormalized posterior
        """
        mean = np.array(self.variational_params[: self.num_parameters])
        covariance = np.diag(
            np.exp(
                2 * np.array(self.variational_params[self.num_parameters : 2 * self.num_parameters])
            ).flatten()
        )
        elbo = float(mvn.entropy(mean.flatten(), covariance) + log_unnormalized_posterior_mean)
        self.iteration_data.add(elbo=elbo)
        self.elbo = elbo

    def _verbose_output(self):
        """Give some informative outputs during the VI iterations."""
        _logger.info("Iteration %s of RPVI algorithm", self.stochastic_optimizer.iteration + 1)

        super()._verbose_output()

        if self.stochastic_optimizer.iteration > 1:
            _logger.debug(
                "Likelihood noise variance: %s", self.model.normal_distribution.covariance
            )

    def _prepare_result_description(self):
        """Create the dictionary for the result pickle file.

        Returns:
            result_description (dict): Dictionary with result summary of the analysis
        """
        result_description = super()._prepare_result_description()
        if self.iteration_data:
            result_description["iteration_data"].update(self.iteration_data.to_dict())
        return result_description

    def evaluate_and_gradient(self, sample_batch):
        """Calculate log-likelihood of observation data and its gradient.

        Evaluation of the likelihood model and its gradient for all inputs of the sample
        batch will trigger the actual forward simulation (can be executed in parallel as
        batch-sequential procedure).

        Args:
            sample_batch (np.array): Sample-batch with samples row-wise

        Returns:
            log_likelihood (np.array): Vector of log-likelihood values for different input samples.
            grad_log_likelihood_x (np.array): Row-wise gradients of log-Likelihood w.r.t. to input
                                              samples.
        """
        # The first samples belong to simulation input
        # get simulation output (run actual forward problem)--> data is saved to DB
        log_likelihood, grad_log_likelihood_x = self.model.evaluate_and_gradient(
            sample_batch.reshape(-1, self.num_parameters)
        )
        self.n_sims += len(sample_batch)
        self.iteration_data.add(
            n_sims=self.n_sims,
            likelihood_variance=self.model.normal_distribution.covariance,
            samples=sample_batch,
        )

        return log_likelihood, grad_log_likelihood_x
