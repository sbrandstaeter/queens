"""Gaussian likelihood."""
import numpy as np

from queens.distributions.normal import NormalDistribution
from queens.models.likelihood_models.likelihood_model import LikelihoodModel
from queens.utils.exceptions import InvalidOptionError
from queens.utils.numpy_utils import add_nugget_to_diagonal


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
        forward_model,
        noise_type,
        noise_value=None,
        nugget_noise_variance=0,
        noise_var_iterative_averaging=None,
        y_obs=None,
        experimental_data_reader=None,
    ):
        """Initialize likelihood model.

        Args:
            forward_model (obj): Forward model on which the likelihood model is based
            noise_type (str): String encoding the type of likelihood noise model:
                                Fixed or MAP estimate with Jeffreys prior
            noise_value (array_like): Likelihood (co)variance value
            nugget_noise_variance (float): Lower bound for the likelihood noise parameter
            noise_var_iterative_averaging (obj): Iterative averaging object
            y_obs (array_like): Vector with observations
            experimental_data_reader (obj): Experimental data reader
        """
        if y_obs is None:
            if experimental_data_reader is None:
                raise InvalidOptionError(
                    "You must either provide 'y_obs' or an "
                    "'experimental_data_reader' for GaussianLikelihood."
                )
            y_obs = experimental_data_reader.get_experimental_data()[0]
        if y_obs is not None and experimental_data_reader is not None:
            Warning(
                "You provided 'y_obs' and 'experimental_data_reader' to GaussianLikelihood. "
                "Only provided 'y_obs' is used."
            )

        super().__init__(forward_model, y_obs)

        y_obs_dim = y_obs.size

        if noise_value is None and noise_type.startswith("fixed"):
            raise InvalidOptionError(f"You have to provide a 'noise_value' for {noise_type}.")

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

        normal_distribution = NormalDistribution(self.y_obs, covariance)

        self.nugget_noise_variance = nugget_noise_variance
        self.noise_type = noise_type
        self.noise_var_iterative_averaging = noise_var_iterative_averaging
        self.normal_distribution = normal_distribution

    def evaluate(self, samples):
        """Evaluate likelihood with current set of input samples.

        Args:
            samples (np.array): Input samples

        Returns:
            log_likelihood (np.array): log-likelihood values per model input
        """
        self.response = self.forward_model.evaluate(samples)
        if self.noise_type.startswith('MAP'):
            self.update_covariance(self.response['mean'])
        log_likelihood = self.normal_distribution.logpdf(self.response['mean'])

        return log_likelihood

    def grad(self, samples, upstream_gradient):
        r"""Evaluate gradient of model w.r.t. current set of input samples.

        Consider current model f(x) with input samples x, and upstream function g(f). The provided
        upstream gradient is :math:`\frac{\partial g}{\partial f}` and the method returns
        :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`.

        Args:
            samples (np.array): Input samples
            upstream_gradient (np.array): Upstream gradient function evaluated at input samples
                                          :math:`\frac{\partial g}{\partial f}`

        Returns:
            gradient (np.array): Gradient w.r.t. current set of input samples
                                 :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`
        """
        # shape convention: num_samples x jacobian_shape
        log_likelihood_grad = self.normal_distribution.grad_logpdf(self.response['mean'])
        upstream_gradient = upstream_gradient * log_likelihood_grad
        gradient = self.forward_model.grad(samples, upstream_gradient)
        return gradient

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
