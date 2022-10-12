import logging
import os

import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import print_summary, set_trainable

from pqueens.regression_approximations.regression_approximation import RegressionApproximation
from pqueens.utils.gpf_utils import extract_block_diag, init_scaler, set_transform_function

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
_logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


class GPflowSVGP(RegressionApproximation):
    """Class for creating Sparse Variational GP regression model based on
    GPFlow.

    Key reference:
        J. Hensman, A. Matthews, and Z. Ghahramani, “Scalable Variational Gaussian Process
        Classification” in Artificial Intelligence and Statistics, Feb. 2015, pp. 351–360.
        https://proceedings.mlr.press/v38/hensman15.html

    Attributes:
        x_train (np.ndarray): training inputs
        y_train (np.ndarray): training outputs
        number_posterior_samples (int): number of posterior samples
        number_input_dimensions (int): dimensionality of random features/input dimension
        model (list): list of GPFlow based stochastic variational GP (SVGP)
        mini_batch_size (int): minibatch size to speed up computation of ELBO
        number_training_iterations (int): number of iterations in optimizer for training
        training_data (list): list of training datasets
        scaler_x (sklearn scaler object): scaler for inputs
        scaler_y (sklearn scaler object): scaler for outputs
        dimension_output (int): dimensionality of the output (quantities of interest)
    """

    def __init__(
        self,
        x_train,
        y_train,
        number_posterior_samples,
        number_input_dimensions,
        model,
        mini_batch_size,
        number_training_iterations,
        training_data,
        scaler_x,
        scaler_y,
        dimension_output,
    ):
        """
        Args:
            x_train (np.ndarray): training inputs
            y_train (np.ndarray): training outputs
            number_posterior_samples (int): number of posterior samples
            number_input_dimensions (int): dimensionality of random features/input dimension
            model (list): list of GPFlow based stochastic variational GP (SVGP)
            mini_batch_size (int): minibatch size to speed up computation of ELBO
            number_training_iterations (int): number of iterations in optimizer for training
            training_data (list): list of training datasets
            scaler_x (sklearn scaler object): scaler for inputs
            scaler_y (sklearn scaler object): scaler for outputs
            dimension_output (int): dimensionality of the output (quantities of interest)
        """

        self.x_train = x_train
        self.y_train = y_train
        self.number_posterior_samples = number_posterior_samples
        self.mini_batch_size = mini_batch_size
        self.number_training_iterations = number_training_iterations
        self.training_data = training_data
        self.number_input_dimensions = number_input_dimensions
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.dimension_output = dimension_output

    @classmethod
    def from_config_create(cls, config, approx_name, x_train, y_train):
        """Create approximation from options dictionary.

        Args:
            config (dict): dictionary with options
            approx_name (str): name of approximation method
            x_train (np.array): training inputs
            y_train (np.array): training outputs

        Returns:
            GPFlowRegression: approximation object
        """
        approx_options = config[approx_name]
        number_posterior_samples = approx_options.get('number_posterior_samples', None)
        seed = approx_options.get('seed', 41)
        mini_batch_size = approx_options.get('mini_batch_size', 100)
        number_training_iterations = approx_options.get('number_training_iterations', 10000)

        np.random.seed(seed)
        tf.random.set_seed(seed)

        (
            training_data,
            x_train,
            y_train,
            scaler_x,
            scaler_y,
            number_input_dimensions,
            dimension_output,
        ) = cls._init_training_dataset(x_train, y_train, seed)
        model = cls._build_model(approx_options, dimension_output, x_train, y_train)

        return cls(
            x_train,
            y_train,
            number_posterior_samples,
            number_input_dimensions,
            model,
            mini_batch_size,
            number_training_iterations,
            training_data,
            scaler_x,
            scaler_y,
            dimension_output,
        )

    def train(self):
        """Train the GP."""
        optimizer = tf.optimizers.Adam()
        for i in range(self.dimension_output):
            training_iterations = iter(self.training_data[i].batch(self.mini_batch_size))
            training_loss = self.model[i].training_loss_closure(training_iterations, compile=True)

            @tf.function
            def optimization_step():
                optimizer.minimize(training_loss, self.model[i].trainable_variables)

            for step in range(self.number_training_iterations):
                optimization_step()
                if (step + 1) % 100 == 0:
                    _logger.info(
                        'Iter: %s/%s, ' 'Loss: %.2e',
                        step + 1,
                        self.number_training_iterations,
                        training_loss().numpy(),
                    )

            print_summary(self.model[i])

    def predict(self, x_test, support='f', full_cov=False):
        """Predict the posterior distribution at x_test.

        Options:
            'f': predict the latent function values
            'y': predict values of the new observations (including noise)

        Args:
            x_test (np.ndarray): new inputs where to make predictions.
            support (str): probabilistic support of random process (default: 'y').
            full_cov (bool): Boolean that specifies whether the entire posterior covariance
                             matrix should be returned or only the posterior variance.

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
                           posterior samples at x_test
        """
        assert support == 'f' or support == 'y', "Unknown input for support."
        x_test = np.atleast_2d(x_test).reshape((-1, self.number_input_dimensions))
        number_test_samples = x_test.shape[0]
        x_test = self.scaler_x.transform(x_test)

        output = {
            'mean': [],
            'variance': [],
            'variance_diagonal': [],
            'x_test': x_test,
            'post_samples': [],
        }

        for i in range(self.dimension_output):
            if support == 'y':
                mean, var = self.model[i].predict_y(x_test, full_cov=False)
            else:
                mean, var = self.model[i].predict_f(x_test, full_cov=full_cov)

            mean = self.scaler_y.inverse_transform(mean.numpy()).reshape((number_test_samples, -1))
            var = var.numpy() * self.scaler_y.var_[i]

            output['mean'].append(mean)
            if support == 'f' and full_cov is True:
                output['variance'].append(np.squeeze(var, axis=0))
                output['variance_diagonal'].append(
                    extract_block_diag(np.squeeze(var, axis=0), output['mean'][-1].shape[1])
                )
            else:
                output['variance'].append(var)

            if self.number_posterior_samples:
                output['post_samples'].append(
                    self.model[i]
                    .predict_f_samples(x_test, self.number_posterior_samples)
                    .numpy()
                    .reshape((x_test.shape[0], self.number_posterior_samples))
                )

        self._squeeze_output(output)

        return output

    @classmethod
    def _init_training_dataset(cls, x_train, y_train, seed):
        """Initialize the training data set.

        Args:
            x_train (np.array): training inputs
            y_train (np.array): training outputs
            seed (int): seed for random number generator

        Returns:
            training_dataset:
            x_train (np.array): training inputs
            y_train (np.array): training outputs
            scaler_x (sklearn scaler object): scaler for inputs
            scaler_y (sklearn scaler object): scaler for outputs
            number_input_dimensions (int): dimensionality of random features/input dimension
            dimension_output (int): number of output dimensions
        """
        if len(x_train.shape) == 1:
            number_input_dimensions = 1
        else:
            number_input_dimensions = x_train.shape[1]

        if len(y_train.shape) != 3:
            y_train = np.expand_dims(y_train, axis=1)
        dimension_output = y_train.shape[1]
        y_train = np.moveaxis(y_train, 1, 2)
        y_train = y_train.reshape(-1, dimension_output)

        scaler_x, x_train = init_scaler(x_train)
        scaler_y, y_train = init_scaler(y_train)

        training_dataset = []
        for i in range(dimension_output):
            training_dataset.append(
                tf.data.Dataset.from_tensor_slices((x_train, y_train[:, [i]]))
                .repeat()
                .shuffle(x_train.shape[0], seed=seed)
            )

        return (
            training_dataset,
            x_train,
            y_train,
            scaler_x,
            scaler_y,
            number_input_dimensions,
            dimension_output,
        )

    @classmethod
    def _build_model(cls, approx_options, dimension_output, x_train, y_train):
        """Build the SVGP model.

        Args:
            approx_options (dict): dictionary with options for approximation method
            dimension_output (int): number of output dimensions
            x_train (np.array): training inputs
            y_train (np.array): training outputs

        Returns:
            model (gpf.models.svgp.SVGP): GPFlow SVGP object
        """
        dimension_lengthscales = approx_options.get('dimension_lengthscales', None)
        lengthscales_0 = 0.1 * np.ones(dimension_lengthscales)
        variances_0 = max(abs(np.max(y_train) - np.min(y_train)), 1e-6)

        train_inducing_points_location = approx_options.get('train_inducing_points_location', False)
        inducing_points = cls._init_inducing_points(x_train, approx_options)

        model = []
        kernel = []
        for i in range(dimension_output):
            kernel.append(gpf.kernels.RBF(lengthscales=lengthscales_0, variance=variances_0))

            model.append(
                gpf.models.SVGP(
                    kernel[i],
                    gpf.likelihoods.Gaussian(),
                    inducing_points,
                    num_data=x_train.shape[0],
                )
            )

            if not train_inducing_points_location:
                gpf.set_trainable(model[i].inducing_variable, False)

            train_likelihood_variance = approx_options.get('train_likelihood_variance', True)
            if not train_likelihood_variance:
                model[i].likelihood.variance.assign(1.1e-6)  # small value for numerical stability
                set_trainable(model[i].likelihood.variance, False)

            model[i].kernel.lengthscales = set_transform_function(
                model[i].kernel.lengthscales,
                tfp.bijectors.Exp(),
            )
            model[i].kernel.variance = set_transform_function(
                model[i].kernel.variance, tfp.bijectors.Exp()
            )

        return model

    @classmethod
    def _init_inducing_points(cls, x_train, approx_options):
        """Initialize inducing points.

        Args:
            x_train (np.ndarray): training inputs
            approx_options (dict): dictionary with options for approximation method

        Returns:
            inducing_points (np.ndarray): inducing points
        """
        number_inducing_points_from_input = approx_options.get('number_inducing_points', 100)
        number_inducing_points = min(x_train.shape[0], number_inducing_points_from_input)
        idx = np.arange(0, x_train.shape[0], 1)
        idx = np.random.choice(idx, size=number_inducing_points, replace=False)
        inducing_points = x_train[idx, :].copy()

        return inducing_points

    @staticmethod
    def _squeeze_output(output):
        """Squeeze output.

        Args:
            output (dict): output dictionary

        Returns:
            output (dict): output dictionary with squeezed arrays
        """

        def _squeeze_array(key):
            if output[key]:
                output[key] = np.squeeze(np.moveaxis(np.array(output[key]), 0, 1), axis=1)

        for current_key in ['mean', 'variance', 'variance_diagonal', 'post_samples']:
            _squeeze_array(current_key)

        return output
