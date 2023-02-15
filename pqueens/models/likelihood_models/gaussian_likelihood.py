"""Gaussian likelihood."""
import numpy as np

from pqueens.distributions import from_config_create_distribution
from pqueens.models.likelihood_models.likelihood_model import LikelihoodModel
from pqueens.utils.gradient_handler import prepare_downstream_gradient_fun
from pqueens.utils.iterative_averaging_utils import from_config_create_iterative_averaging
from pqueens.utils.numpy_utils import add_nugget_to_diagonal


class GaussianLikelihood(LikelihoodModel):
    r"""Gaussian likelihood model with fixed or dynamic noise.

    The noise can be modelled by a full covariance matrix, independent variances or a unified
    variance for all observations. If the noise is chosen to be dynamic, a MAP estimate of the
    covariance, independent variances or unified variance is computed using a Jeffrey's prior.
    Jeffrey's prior is defined as :math:`\pi_J(\Sigma) = |\Sigma|^{-(p+2)/2}`, where :math:`\Sigma`
    is the covariance matrix of shape :math:`p \times p` (see [1])

    References:
        [1]: Sun, Dongchu, and James O. Berger. "Objective Bayesian analysis for the multivariate
             normal model." Bayesian Statistics 8 (2007): 525-562.

    Attributes:
        nugget_noise_variance (float): Lower bound for the likelihood noise parameter
        noise_type (str): String encoding the type of likelihood noise model:
                                     Fixed or MAP estimate with Jeffreys prior
        noise_var_iterative_averaging (obj): Iterative averaging object
        normal_distribution (obj): Underlying normal distribution object

    Returns:
        Instance of GaussianLikelihood Class
    """

    def __init__(
        self,
        model_name,
        nugget_noise_variance,
        forward_model,
        noise_type,
        noise_var_iterative_averaging,
        normal_distribution,
        coords_mat,
        time_vec,
        y_obs,
        output_label,
        coord_labels,
    ):
        """Initialize likelihood model.

        Args:
            model_name (str): Model name
            forward_model (obj): Forward model on which the likelihood model is based
            nugget_noise_variance (float): Lower bound for the likelihood noise parameter
            noise_type (str): String encoding the type of likelihood noise model:
                                         Fixed or MAP estimate with Jeffreys prior
            noise_var_iterative_averaging (obj): Iterative averaging object
            normal_distribution (obj): Underlying normal distribution object
            coords_mat (np.array): Matrix of observation coordinates (new coordinates row-wise)
            time_vec (np.array): Vector containing time stamps for each observation
            y_obs (np.array): Matrix with row-wise observation vectors
            output_label (str): Output label name of the observations
            coord_labels (list): List of coordinate label names. One name per column in coord_mat
        """
        super().__init__(
            model_name,
            forward_model,
            coords_mat,
            time_vec,
            y_obs,
            output_label,
            coord_labels,
        )
        self.nugget_noise_variance = nugget_noise_variance
        self.noise_type = noise_type
        self.noise_var_iterative_averaging = noise_var_iterative_averaging
        self.normal_distribution = normal_distribution

    @classmethod
    def from_config_create_model(
        cls,
        model_name,
        config,
    ):
        """Create Gaussian likelihood model from problem description.

        Args:
            model_name (str): Name of the likelihood model
            config (dict): Dictionary containing problem description

        Returns:
            instance of GaussianLikelihood class
        """
        (
            forward_model,
            coords_mat,
            time_vec,
            y_obs,
            output_label,
            coord_labels,
        ) = super().get_base_attributes_from_config(model_name, config)
        y_obs_dim = y_obs.size

        # get options
        model_options = config[model_name]

        # get specifics of gaussian likelihood model
        noise_type = model_options["noise_type"]
        noise_value = model_options.get("noise_value")
        nugget_noise_variance = model_options.get("nugget_noise_variance", 1e-6)

        noise_var_iterative_averaging = model_options.get("noise_var_iterative_averaging", None)
        if noise_var_iterative_averaging:
            noise_var_iterative_averaging = from_config_create_iterative_averaging(
                noise_var_iterative_averaging
            )

        if noise_type == 'fixed_variance':
            covariance = noise_value * np.eye(y_obs_dim)
        elif noise_type == 'fixed_variance_vector':
            covariance = np.diag(noise_value)
        elif noise_type == 'fixed_covariance_matrix':
            covariance = noise_value
        elif noise_type in [
            'MAP_jeffrey_variance',
            'MAP_jeffrey_variance_vector',
            'MAP_jeffrey_covariance_matrix',
        ]:
            covariance = np.eye(y_obs_dim)
        else:
            raise NotImplementedError

        distribution_options = {"type": "normal", "mean": y_obs, "covariance": covariance}
        normal_distribution = from_config_create_distribution(distribution_options)
        return cls(
            model_name=model_name,
            nugget_noise_variance=nugget_noise_variance,
            forward_model=forward_model,
            noise_type=noise_type,
            noise_var_iterative_averaging=noise_var_iterative_averaging,
            normal_distribution=normal_distribution,
            coords_mat=coords_mat,
            time_vec=time_vec,
            y_obs=y_obs,
            output_label=output_label,
            coord_labels=coord_labels,
        )

    def evaluate(self, samples):
        """Evaluate likelihood with current set of samples.

        Args:
            samples (np.array): Evaluated samples

        Returns:
            log_likelihood (np.array): Vector of log-likelihood values
                                       per model input.
        """
        forward_model_output = self.forward_model.evaluate(samples)["mean"]
        log_likelihood = self.evaluate_from_output(forward_model_output)
        return log_likelihood

    def evaluate_from_output(self, forward_model_output):
        """Evaluate likelihood with forward model output given.

        Args:
            forward_model_output (np.array): Evaluated forward
                                             model output

        Returns:
            log_likelihood (np.array): Vector of log-likelihood values
                                       per model output.
        """
        if self.noise_type.startswith('MAP'):
            self.update_covariance(forward_model_output)
        log_likelihood = self.normal_distribution.logpdf(forward_model_output)

        return log_likelihood

    def update_covariance(self, y_model):
        """Update covariance matrix of the gaussian likelihood.

        Args:
            y_model (np.ndarray): Forward model output with shape (samples, outputs)
        """
        dist = y_model - self.y_obs.reshape(1, -1)
        num_samples, dim_y = y_model.shape
        if self.noise_type == 'MAP_jeffrey_variance':
            covariance = np.eye(dim_y) / (dim_y * (num_samples + dim_y + 2)) * np.sum(dist**2)
        elif self.noise_type == 'MAP_jeffrey_variance_vector':
            covariance = np.diag(1 / (num_samples + dim_y + 2) * np.sum(dist**2, axis=0))
        else:
            covariance = 1 / (num_samples + dim_y + 2) * np.dot(dist.T, dist)

        # If iterative averaging is desired
        if self.noise_var_iterative_averaging:
            covariance = self.noise_var_iterative_averaging.update_average(covariance)

        covariance = add_nugget_to_diagonal(covariance, self.nugget_noise_variance)
        self.normal_distribution.update_covariance(covariance)

    def evaluate_and_gradient(self, samples, upstream_gradient_fun=None):
        """Evaluate likelihood model and gradient with current set of samples.

        Args:
            samples (np.array): Current input samples for which likelihood should be evaluated
            upstream_gradient_fun (function): The gradient of an upstream objective function w.r.t.
                                              the model output. The expected input arguments for
                                              this function are 1) the model input samples and 2)
                                              the model output corresponding to the input samples.

        Returns:
            log_likelihood (np.array): Vector of log-likelihood values for different input samples.
            gradient_objective_fun_samples (np.array): Row-wise gradients of the objective function
                                                       w.r.t. to the input samples. If the the
                                                       method argument 'grad_objective_fun' is
                                                       None, the objective function  is the
                                                       evaluation function of this model, the
                                                       likelihood function, itself.
        """
        # compose the gradient objective function to update it with own partial derivative
        downstream_gradient_fun = prepare_downstream_gradient_fun(
            eval_output_fun=self.evaluate_from_output,
            partial_grad_evaluate_fun=self.partial_grad_evaluate,
            upstream_gradient_fun=upstream_gradient_fun,
        )
        # call evaluate_and_gradient of sub model with the downstream gradient function
        # (which is the upstream gradient function from the perspective of the sub model)
        sub_model_output, gradient_objective_fun_samples = self.forward_model.evaluate_and_gradient(
            samples, upstream_gradient_fun=downstream_gradient_fun
        )

        # evaluate log-likelihood reusing the sub model evaluations
        log_likelihood = self.evaluate_from_output(sub_model_output)

        return log_likelihood, gradient_objective_fun_samples

    def partial_grad_evaluate(self, _forward_model_input, forward_model_output):
        """Implement the partial derivative of the evaluate method.

        The partial derivative w.r.t. the output of the sub-model is for example
        required to calculate gradients of the current model w.r.t. to the sample
        input.

        Args:
            _forward_model_input (np.array): Sample inputs of the model run (here not required).
            forward_model_output (np.array): Output of the underlying sub- or forward model
                                             for the current batch of sample inputs.

        Returns:
            grad_out (np.array): Evaluated partial derivative of the evaluation function
                                 w.r.t. the output of the underlying sub-model.
        """
        grad_out = self.normal_distribution.grad_logpdf(forward_model_output)
        return grad_out
