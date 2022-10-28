import logging

import GPy
import numpy as np
from sklearn.preprocessing import StandardScaler

from pqueens.regression_approximations.regression_approximation import RegressionApproximation
from pqueens.regression_approximations.utils.gpy_kernels import get_gpy_kernel_type
from pqueens.utils.logger_settings import log_multiline_string

_logger = logging.getLogger(__name__)


class GPGPyRegression(RegressionApproximation):
    """Class for creating GP based regression model based on GPy.

    This class constructs a GP regression using a GPy model.

    Attributes:
        x_train (np.array): Training inputs
        y_train (np.array): Training outputs
        scaler_x (sklearn scaler object): Scaler for inputs
        number_posterior_samples (int): Number of posterior samples
        model (Gpy.model): GPy based Gaussian process model
        seed_optimizer (int): seed for optimizer for training the GP
        seed_posterior_samples (int): seed for posterior samples
    """

    def __init__(
        self,
        x_train,
        y_train,
        scaler_x,
        number_posterior_samples,
        number_input_dimensions,
        model,
        seed_optimizer,
        seed_posterior_samples,
        number_restarts,
        number_optimizer_iterations,
    ):

        self.x_train = x_train
        self.y_train = y_train
        self.scaler_x = scaler_x
        self.number_posterior_samples = number_posterior_samples
        self.number_input_dimensions = number_input_dimensions
        self.model = model
        self.seed_optimizer = seed_optimizer
        self.seed_posterior_samples = seed_posterior_samples
        self.number_restarts = number_restarts
        self.number_optimizer_iterations = number_optimizer_iterations

    @classmethod
    def from_config_create(cls, config, approx_name, x_train, y_train):
        """Create approximation from options dictionary.

        Args:
            config (dict):         Dictionary with problem description (input file)
            approx_name (str):     Name of approximation method
            x_train (np.array):    Training inputs
            y_train (np.array):    Training outputs

        Returns:
            gp_approximation_gpy: approximation object
        """
        approx_options = config[approx_name]
        number_posterior_samples = approx_options.get('num_posterior_samples', None)
        ard = approx_options.get('ard', False)
        kernel_type = approx_options.get('kernel_type', "sum_rbf")
        seed_optimizer = approx_options.get("seed_optimizer", 42)
        seed_posterior_samples = approx_options.get("seed_posterior_samples", None)
        number_restarts = approx_options.get("num_restart", 5)
        number_optimizer_iterations = approx_options.get("number_optimizer_iterations", 1000)

        normalize_y = approx_options.get("normalize_y", True)

        # input dimension
        if len(x_train.shape) == 1:
            number_input_dimensions = 1
        else:
            number_input_dimensions = x_train.shape[1]

        if y_train.ndim == 1:
            y_train = y_train.reshape((-1, 1))

        # scaling of input dimensions
        scaler_x = StandardScaler()
        scaler_x.fit(x_train)
        x_train = scaler_x.transform(x_train)

        lengthscale_0 = 0.1 * abs(np.max(x_train) - np.min(x_train))

        if normalize_y is True:
            variance_0 = 1.0
        else:
            variance_0 = abs(np.max(y_train) - np.min(y_train))

        kernel = cls._setup_kernel(
            kernel_type, number_input_dimensions, variance_0, lengthscale_0, ard
        )
        model = GPy.models.GPRegression(
            x_train,
            y_train,
            kernel=kernel,
            normalizer=normalize_y,
        )
        log_multiline_string(_logger, str(model))

        return cls(
            x_train,
            y_train,
            scaler_x,
            number_posterior_samples,
            number_input_dimensions,
            model,
            seed_optimizer,
            seed_posterior_samples,
            number_restarts,
            number_optimizer_iterations,
        )

    def train(self):
        """Train the GP by maximizing the likelihood."""
        # fix seed for randomize in restarts
        np.random.seed(self.seed_optimizer)
        self.model.optimize_restarts(
            num_restarts=self.number_restarts,
            max_iters=self.number_optimizer_iterations,
            messages=True,
        )
        log_multiline_string(_logger, str(self.model))

    def predict(self, x_test, support='y', full_cov=False):
        """Predict the posterior distribution at x_test with respect to the
        data 'y' or the latent function 'f'.

        Args:
            x_test (np.array): Inputs at which to evaluate latent function f
            support (str): Probabilistic support of random process (default: 'y'). Possible options
                           are 'y' or 'f'. Here, 'f' means the latent function so that the posterior
                           variance of the GP is calculated with respect to f. In contrast 'y'
                           refers to the data itself so that the posterior variance is computed
                           with respect to 'y' (f is integrated out) leading to an extra addition
                           of noise in the posterior variance.
            full_cov (bool): Boolean that specifies whether the entire posterior covariance matrix
                             should be returned or only the posterior variance

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
                           posterior samples at x_test
        """
        x_test = np.atleast_2d(x_test).reshape((-1, self.model.input_dim))
        x_test = self.scaler_x.transform(x_test)

        if support == 'y':
            output = self.predict_y(x_test, full_cov=full_cov)
        elif support == 'f':
            output = self.predict_f(x_test, full_cov=full_cov)
        else:
            raise NotImplementedError(
                f"You choose support={support} but the only valid options are 'y' or 'f'"
            )
        if self.number_posterior_samples is not None:
            output["post_samples"] = self.predict_f_samples(x_test, self.number_posterior_samples)
        return output

    def predict_y(self, x_test, full_cov=False):
        """Compute the posterior distribution at x_test with respect to the
        data 'y'.

        Args:
            x_test (np.array): Inputs at which to evaluate latent function f
            full_cov (bool): Boolean that specifies whether the entire posterior covariance
                             matrix should be returned or only the posterior variance.

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
                           posterior samples at x_test
        """
        output = {"x_test": x_test}
        output["mean"], output["variance"] = self.model.predict(x_test, full_cov=full_cov)
        if self.number_posterior_samples is not None:
            output["post_samples"] = self.predict_f_samples(x_test, self.number_posterior_samples)

        return output

    def predict_f(self, x_test, full_cov=False):
        """Compute the mean and variance of the latent function at x_test.

        Args:
            x_test (np.array): Inputs at which to evaluate latent function f
            full_cov (bool): Boolean that specifies whether the entire posterior covariance
                             matrix should be returned or only the posterior variance.

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
                           posterior samples at x_test
        """
        output = {"x_test": x_test}
        output["mean"], output["variance"] = self.model.predict_noiseless(x_test, full_cov=full_cov)

        return output

    def predict_f_samples(self, x_test, num_samples):
        """Produce samples from the posterior latent function x_test.

        Args:
            x_test (np.array):    Inputs at which to evaluate latent function f
            num_samples (int):  Number of posterior field_realizations of GP

        Returns:
            np.array, np.array: mean and variance of latent functions at x_test
        """
        if self.seed_posterior_samples:
            # fix seed for random samples
            np.random.seed(self.seed_posterior_samples)
            _logger.warning("Beware, the seed for drawing posterior samples is fixed.")
        post_samples = self.model.posterior_samples_f(x_test, num_samples)
        # GPy returns 3d array middle dimension indicates number of outputs, i.e.
        # it is only != 1 for multi-output processes
        if post_samples.shape[1] != 1:
            raise Exception("GPGPyRegression can not deal with multi-output GPs")
        return np.reshape(post_samples, (x_test.shape[0], num_samples))

    @classmethod
    def _setup_kernel(cls, kernel_type, input_dim, variance_0, lengthscale_0, ard):
        """Choose the correct kernel setup method and setup kernel.

        Args:
            kernel_type (str): kernel type to setup
            input_dim (int): number of input dimensions
            variance_0 (float): initial kernel variance of the GP
            lengthscale_0 (float): initial kernel lengthscale(s) of the GP
            ard (bool): true if automatic relevance determination is active

        Returns:
            kernel (GPy.kern object): kernel for Gaussian Process
        """

        setup_specific_kernel = get_gpy_kernel_type(kernel_type)
        kernel = setup_specific_kernel(input_dim, variance_0, lengthscale_0, ard)
        log_multiline_string(_logger, str(kernel))

        return kernel
