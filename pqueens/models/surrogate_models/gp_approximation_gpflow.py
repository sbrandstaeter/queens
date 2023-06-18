"""Gaussian process implementation in GPFlow."""

import logging
import os

import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import print_summary, set_trainable

from pqueens.regression_approximations.regression_approximation import RegressionApproximation
from pqueens.utils.gpf_utils import extract_block_diag, init_scaler, set_transform_function

_logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress warnings

# Use GPU acceleration
if tf.test.gpu_device_name() != '/device:GPU:0':
    _logger.info('WARNING: GPU device not found.')
else:
    _logger.info('SUCCESS: Found GPU: %s', tf.test.gpu_device_name())


class GPFlowRegression(RegressionApproximation):
    """Class for creating GP regression model based on GPFlow.

    This class constructs a GP regression, using a GPFlow model.

    Attributes:
        x_train (np.ndarray): Training inputs.
        y_train (np.ndarray): Training outputs.
        number_posterior_samples (int): Number of posterior samples.
        number_restarts (int): Number of restarts.
        number_training_iterations (int): Number of iterations in optimizer for training.
        number_input_dimensions (int): Dimensionality of random features.
        restart_min_value (int): Minimum value for restart.
        restart_max_value (int): Maximum value for restart.
        model (GPFlow.models.GPR): GPFlow based Gaussian process model.
        dimension_lengthscales (int): Dimension of *lengthscales*.
        scaler_x (sklearn scaler object): Scaler for inputs.
        scaler_y (sklearn scaler object): Scaler for outputs.
    """

    def __init__(
        self,
        x_train,
        y_train,
        number_posterior_samples,
        number_input_dimensions,
        restart_min_value,
        restart_max_value,
        model,
        number_restarts,
        number_training_iterations,
        dimension_lengthscales,
        scaler_x,
        scaler_y,
    ):
        """TODO_doc.

        Args:
            x_train (np.ndarray): Training inputs
            y_train (np.ndarray): Training outputs
            number_posterior_samples (int): Number of posterior samples
            number_input_dimensions (int): Dimensionality of random features/input dimension
            restart_min_value (int): Minimum value for restart
            restart_max_value (int): Maximum value for restart
            model (GPFlow.models.GPR): GPFlow based Regression model
            number_restarts (int): Number of restarts
            number_training_iterations (int): Number of iterations in optimizer for training
            dimension_lengthscales (int): Dimension of lengthscales
            scaler_x (sklearn scaler object): Scaler for inputs
            scaler_y (sklearn scaler object): Scaler for outputs
        """
        self.x_train = x_train
        self.y_train = y_train
        self.number_posterior_samples = number_posterior_samples
        self.number_restarts = number_restarts
        self.number_training_iterations = number_training_iterations
        self.number_input_dimensions = number_input_dimensions
        self.restart_min_value = restart_min_value
        self.restart_max_value = restart_max_value
        self.model = model
        self.dimension_lengthscales = dimension_lengthscales
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

    @classmethod
    def from_config_create(cls, config, approx_name, x_train, y_train):
        """Create approximation from options dictionary.

        Args:
            config (dict):         Dictionary with options
            approx_name (str):     Name of approximation method
            x_train (np.array):    Training inputs
            y_train (np.array):    Training outputs

        Returns:
            GPFlowRegression: Approximation object
        """
        number_posterior_samples = config[approx_name].get('number_posterior_samples', None)

        if len(x_train.shape) == 1:
            number_input_dimensions = 1
        else:
            number_input_dimensions = x_train.shape[1]

        if y_train.shape[0] != x_train.shape[0]:
            y_train = np.reshape(y_train, (-1, 1))

        number_restarts = config[approx_name].get('number_restarts', 10)
        number_training_iterations = config[approx_name].get('number_training_iterations', 100)
        restart_min_value = config[approx_name].get('restart_min_value', 0)
        restart_max_value = config[approx_name].get('restart_max_value', 5)

        scaler_x, x_train = init_scaler(x_train)
        scaler_y, y_train = init_scaler(y_train)

        # initialize hyperparameters
        dimension_lengthscales = config[approx_name].get('dimension_lengthscales', None)
        lengthscales_0 = 0.1 * np.ones(dimension_lengthscales)
        variances_0 = max(abs(np.max(y_train) - np.min(y_train)), 1e-6)

        # choose kernel
        kernel = gpf.kernels.RBF(lengthscales=lengthscales_0, variance=variances_0)

        # initialize model
        model = gpf.models.GPR(data=(x_train, y_train), kernel=kernel, mean_function=None)

        train_likelihood_variance = config[approx_name].get('train_likelihood_variance', True)
        if not train_likelihood_variance:
            model.likelihood.variance.assign(1.1e-6)  # small value for numerical stability
            set_trainable(model.likelihood.variance, False)

        model.kernel.lengthscales = set_transform_function(
            model.kernel.lengthscales, tfp.bijectors.Exp()
        )
        model.kernel.variance = set_transform_function(model.kernel.variance, tfp.bijectors.Exp())

        return cls(
            x_train,
            y_train,
            number_posterior_samples,
            number_input_dimensions,
            restart_min_value,
            restart_max_value,
            model,
            number_restarts,
            number_training_iterations,
            dimension_lengthscales,
            scaler_x,
            scaler_y,
        )

    def train(self):
        """Train the GP by maximizing the likelihood."""
        opt = gpf.optimizers.Scipy()

        dimension_hyperparameters = self.get_dimension_hyperparameters()
        loss = np.ones([self.number_restarts + 1])
        train_logs = []
        for i in range(self.number_restarts + 1):
            if i > 0:
                hyperparameters = tf.random.uniform(
                    [dimension_hyperparameters],
                    minval=self.restart_min_value,
                    maxval=self.restart_max_value,
                    seed=i,
                )
                self.assign_hyperparameters(hyperparameters, transform=False)
            try:
                opt_logs = opt.minimize(
                    self.model.training_loss,
                    self.model.trainable_variables,
                    options={"maxiter": self.number_training_iterations},
                )
                loss[i] = opt_logs.fun
                train_logs.append(opt_logs)
            except tf.errors.InvalidArgumentError:
                loss[i] = np.nan
                train_logs.append('Optimization Failed')
            _logger.info('restart %s/%s    loss = %s', i, self.number_restarts, loss[i])

        hyperparameters = train_logs[int(np.nanargmin(loss))].x

        self.assign_hyperparameters(hyperparameters, transform=True)

        print_summary(self.model)

    def predict(self, x_test, support='y', full_cov=False):
        """Predict the posterior distribution at *x_new*.

        Options:

            - *f(x_test)*: predict the latent function values
            - *y(x_test)*: predict values of the new observations (including noise)

        Args:
            x_test (np.ndarray): New inputs where to make predictions
            support (str): Probabilistic support of random process (default: 'y')
            full_cov (bool): Boolean that specifies whether the entire posterior covariance
                             matrix should be returned or only the posterior variance

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
            posterior samples at *x_test*
        """
        x_test = np.atleast_2d(x_test).reshape((-1, self.number_input_dimensions))
        number_test_samples = x_test.shape[0]
        x_test = self.scaler_x.transform(x_test)

        if support == 'y':
            mean, var = self.model.predict_y(x_test, full_cov=False)
        elif support == 'f':
            mean, var = self.model.predict_f(x_test, full_cov=full_cov)
        else:
            mean = None
            var = None

        mean = self.scaler_y.inverse_transform(mean.numpy())
        var = var.numpy() * self.scaler_y.var_

        output = {'mean': mean.reshape(number_test_samples, -1), 'x_test': x_test}
        if support == 'f' and full_cov is True:
            output['variance'] = np.squeeze(var, axis=0)
            output['variance_diagonal'] = extract_block_diag(
                np.squeeze(var, axis=0), output['mean'].shape[1]
            )
        else:
            output['variance'] = var

        if self.number_posterior_samples:
            output['post_samples'] = (
                self.model.predict_f_samples(x_test, self.number_posterior_samples)
                .numpy()
                .reshape((x_test.shape[0], self.number_posterior_samples))
            )

        return output

    def assign_hyperparameters(self, hyperparameters, transform=False):
        """Assign untransformed (constrained) hyperparameters to model.

        Args:
            hyperparameters (np.ndarray):   Hyperparameters of GP
            transform (bool):               If *True*, hyperparameters are transformed from
                                            unconstrained to constrained representation
        """
        hyperparameters = tf.convert_to_tensor(hyperparameters)
        if transform:
            hyperparameters = self.transform_hyperparameters(hyperparameters)

        self.model.kernel.lengthscales.assign(hyperparameters[0 : self.dimension_lengthscales])
        self.model.kernel.variance.assign(hyperparameters[self.dimension_lengthscales])

    def transform_hyperparameters(self, hyperparameters):
        """TODO_doc: add a one-line explanation.

        Transform hyperparameters from unconstrained to constrained
        representation.

        Args:
            hyperparameters (np.ndarray):   Unconstrained representation of hyperparameters
        Returns:
            hyperparameters (np.ndarray): Constrained representation of hyperparameters
        """
        hyperparameters = tf.convert_to_tensor(hyperparameters)

        lengthscales = self.model.kernel.lengthscales.transform.forward(
            hyperparameters[0 : self.dimension_lengthscales]
        )
        variances = tf.reshape(
            self.model.kernel.variance.transform.forward(
                hyperparameters[self.dimension_lengthscales]
            ),
            [-1],
        )

        hyperparameters = tf.concat([lengthscales, variances], 0)

        return hyperparameters

    def get_dimension_hyperparameters(self):
        """Return the dimension of the hyperparameters.

        Returns:
            dimension_hyperparameters (int): Dimension of hyperparameters
        """
        lengthscales = self.model.kernel.lengthscales.unconstrained_variable
        variances = tf.reshape(self.model.kernel.variance.unconstrained_variable, [-1])
        hyperparameters = tf.concat([lengthscales, variances], 0)

        dimension_hyperparameters = hyperparameters.shape.dims[0]

        return dimension_hyperparameters
