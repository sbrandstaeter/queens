"""TODO_doc."""

import logging

import GPy
import numpy as np
from sklearn.preprocessing import StandardScaler

from pqueens.iterators import from_config_create_iterator
from pqueens.models.surrogate_models.surrogate_model import SurrogateModel
from pqueens.models.surrogate_models.utils.gpy_kernels import get_gpy_kernel_type

_logger = logging.getLogger(__name__)


class GPGPyRegressionModel(SurrogateModel):
    """Class for creating GP based regression model based on GPy.

    This class constructs a GP regression using a GPy model.

    Attributes:
        x_train (np.array): Training inputs.
        y_train (np.array): Training outputs.
        scaler_x (sklearn scaler object): Scaler for inputs.
        number_posterior_samples (int): Number of posterior samples.
        number_input_dimensions: TODO_doc
        model (Gpy.model): GPy based Gaussian process model.
        seed_optimizer (int): Seed for optimizer for training the GP.
        seed_posterior_samples (int): Seed for posterior samples.
        number_restarts: TODO_doc
        number_optimizer_iterations: TODO_doc
    """

    def __init__(
        self,
        training_iterator,
        testing_iterator=None,
        eval_fit=None,
        error_measures=None,
        nash_sutcliffe_efficiency=False,
        plotting_options=None,
        num_posterior_samples=None,
        ard=False,
        kernel_type="sum_rbf",
        seed_optimizer=42,
        seed_posterior_samples=None,
        num_restart=5,
        number_optimizer_iterations=1000,
        normalize_y=True,
    ):
        """TODO_doc.

        Args:
            num_posterior_samples: TODO_doc
            seed_optimizer: TODO_doc
            seed_posterior_samples: TODO_doc
            num_restart: TODO_doc
            number_optimizer_iterations: TODO_doc
        """
        super().__init__(
            training_iterator=training_iterator,
            testing_iterator=testing_iterator,
            eval_fit=eval_fit,
            error_measures=error_measures,
            nash_sutcliffe_efficiency=nash_sutcliffe_efficiency,
            plotting_options=plotting_options,
        )
        self.x_train = None
        self.y_train = None
        self.scaler_x = None
        self.number_posterior_samples = num_posterior_samples
        self.number_input_dimensions = None
        self.model = None
        self.seed_optimizer = seed_optimizer
        self.seed_posterior_samples = seed_posterior_samples
        self.number_restarts = num_restart
        self.number_optimizer_iterations = number_optimizer_iterations
        self.ard = ard
        self.kernel_type = kernel_type
        self.normalize_y = normalize_y

    def train(self, x_train, y_train):
        """Train the GP by maximizing the likelihood."""
        # input dimension
        if len(x_train.shape) == 1:
            self.number_input_dimensions = 1
        else:
            self.number_input_dimensions = x_train.shape[1]

        if y_train.ndim == 1:
            y_train = y_train.reshape((-1, 1))
        self.y_train = y_train

        # scaling of input dimensions
        self.scaler_x = StandardScaler()
        self.scaler_x.fit(x_train)
        self.x_train = self.scaler_x.transform(x_train)

        lengthscale_0 = 0.1 * abs(np.max(self.x_train) - np.min(self.x_train))

        if self.normalize_y is True:
            variance_0 = 1.0
        else:
            variance_0 = abs(np.max(self.y_train) - np.min(self.y_train))

        kernel = self._setup_kernel(
            self.kernel_type, self.number_input_dimensions, variance_0, lengthscale_0, self.ard
        )
        self.model = GPy.models.GPRegression(
            self.x_train,
            self.y_train,
            kernel=kernel,
            normalizer=self.normalize_y,
        )
        _logger.info(str(self.model))

        # fix seed for randomize in restarts
        np.random.seed(self.seed_optimizer)
        self.model.optimize_restarts(
            num_restarts=self.number_restarts,
            max_iters=self.number_optimizer_iterations,
            messages=True,
        )
        _logger.info(str(self.model))

    def predict(self, x_test, support='y', full_cov=False):
        """TODO_doc: add a one-line explanation.

        Predict the posterior distribution at *x_test* w.r.t. the data 'y'
        or the latent function 'f'.

        Args:
            x_test (np.array): Inputs at which to evaluate latent function 'f'
            support (str): Probabilistic support of random process (default: 'y'). Possible options
                           are 'y' or 'f'. Here, 'f' means the latent function, so that the
                           posterior variance of the GP is calculated with respect to 'f'. In
                           contrast, 'y' refers to the data itself so that the posterior variance
                           is computed with respect to 'y' ('f' is integrated out), leading to
                           an extra addition of noise in the posterior variance
            full_cov (bool): Boolean that specifies whether the entire posterior covariance matrix
                             should be returned or only the posterior variance

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
            posterior samples at *x_test*
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
        """TODO_doc: add a one-line explanation.

        Compute the posterior distribution at *x_test* with respect to the
        data 'y'.

        Args:
            x_test (np.array): Inputs at which to evaluate latent function 'f'
            full_cov (bool): Boolean that specifies whether the entire posterior covariance
                             matrix should be returned or only the posterior variance

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
            posterior samples at *x_test*
        """
        output = {"x_test": x_test}
        output["mean"], output["variance"] = self.model.predict(x_test, full_cov=full_cov)
        if self.number_posterior_samples is not None:
            output["post_samples"] = self.predict_f_samples(x_test, self.number_posterior_samples)

        return output

    def predict_f(self, x_test, full_cov=False):
        """Compute the mean and variance of the latent function at *x_test*.

        Args:
            x_test (np.array): Inputs at which to evaluate latent function 'f'
            full_cov (bool): Boolean that specifies whether the entire posterior covariance
                             matrix should be returned or only the posterior variance

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
            posterior samples at *x_test*
        """
        output = {"x_test": x_test}
        output["mean"], output["variance"] = self.model.predict_noiseless(x_test, full_cov=full_cov)

        return output

    def predict_f_samples(self, x_test, num_samples):
        """Produce samples from the posterior latent function *x_test*.

        Args:
            x_test (np.array):    Inputs at which to evaluate latent function 'f'
            num_samples (int):  Number of posterior *field_realizations* of GP

        Returns:
            np.array, np.array: Mean and variance of latent functions at *x_test*
        """
        if self.seed_posterior_samples:
            # fix seed for random samples
            np.random.seed(self.seed_posterior_samples)
            _logger.warning("Beware, the seed for drawing posterior samples is fixed.")
        post_samples = self.model.posterior_samples_f(x_test, num_samples)
        # GPy returns 3d array middle dimension indicates number of outputs, i.e.
        # it is only != 1 for multi-output processes
        if post_samples.shape[1] != 1:
            raise ValueError("GPGPyRegression can not deal with multi-output GPs")
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
        _logger.info(str(kernel))

        return kernel
