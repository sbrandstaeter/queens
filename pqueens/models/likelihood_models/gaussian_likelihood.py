"""Gaussian likelihood."""

import logging

import numpy as np

from pqueens.distributions import from_config_create_distribution
from pqueens.models.likelihood_models.likelihood_model import LikelihoodModel
from pqueens.utils.iterative_averaging_utils import from_config_create_iterative_averaging
from pqueens.utils.numpy_utils import add_nugget_to_diagonal

_logger = logging.getLogger(__name__)


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
        model_parameters,
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
            model_parameters (dict): Dictionary with description of uncertain
                                     parameters
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
            model_parameters,
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
    def from_config_create_likelihood(
        cls,
        model_name,
        config,
        model_parameters,
        forward_model,
        coords_mat,
        time_vec,
        y_obs,
        output_label,
        coord_labels,
    ):
        """Create Gaussian likelihood model from problem description.

        Args:
            model_name (str): Name of the likelihood model
            config (dict): Dictionary containing problem description
            model_parameters (dict): Dictionary containing description of model parameters
            forward_model (obj): Forward model on which the likelihood model is based
            coords_mat (np.array): Row-wise coordinates at which the observations were recorded
            time_vec (np.array): Vector of observation times
            y_obs (np.array): Corresponding experimental data vector to coords_mat
            output_label (str): Name of the experimental outputs (column label in csv-file)
            coord_labels (lst): List with coordinate labels for (column labels in csv-file)

        Returns:
            instance of GaussianLikelihood class
        """
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

        y_obs_dim = y_obs.size
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

        distribution_options = {"distribution": "normal", "mean": y_obs, "covariance": covariance}
        normal_distribution = from_config_create_distribution(distribution_options)

        return cls(
            model_name=model_name,
            model_parameters=model_parameters,
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

    def evaluate(self, gradient=False):
        """Evaluate likelihood with current set of samples.

        Returns:
            log_likelihood_output (tuple): Tuple with vector of log-likelihood values
                                            per model input and potentially the gradient
                                            of the model w.r.t. its inputs
            gradient (bool, optional): Boolean to determine whether the gradient of the
                                        likelihood should be evaluated (if set to True)
        """
        model_output = self._update_and_evaluate_forward_model(gradient_bool=gradient)

        if gradient:
            self.update_covariance(model_output[0])
            log_likelihood_output = self.normal_distribution.logpdf(model_output[0])

            grad_y = model_output[1]
            grad_log_likelihood = np.dot(
                self.normal_distribution.grad_logpdf(model_output[0]), grad_y
            )
            log_likelihood_output = (log_likelihood_output, grad_log_likelihood)
        else:
            self.update_covariance(model_output)
            log_likelihood_output = self.normal_distribution.logpdf(model_output)

        return log_likelihood_output

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

    def _update_and_evaluate_forward_model(self, gradient_bool=False):
        """Evaluate forward model.

        Pass the variables update to subordinate model and then evaluate the forward model.

        Args:
            gradient_bool (bool): Flag to determine, whether the gradient of the function at
                                  the evaluation point is expected (True) or not (False)

        Returns:
           model_output (tuple, np.array): Tuple with forward model output with shape
                                          (samples, outputs) and gradient of the model
                                           w.r.t. the model input; or only model output,
                                           if not gradient is required
        """
        # Note that the wrapper of the model update needs to called externally such that
        # self.variables is updated
        self.forward_model.variables = self.variables
        n_samples_batch = len(self.variables)
        if gradient_bool:
            model_output_dict = self.forward_model.evaluate(gradient_bool=gradient_bool)
            model_response = model_output_dict['mean'][-n_samples_batch:]
            model_gradient = model_output_dict['gradient'][
                -n_samples_batch * model_response.shape[1] :, :
            ]
            model_output = (model_response, model_gradient)
        else:
            model_output = self.forward_model.evaluate()['mean'][-n_samples_batch:]

        return model_output
