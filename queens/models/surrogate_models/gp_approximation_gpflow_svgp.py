"""TODO_doc."""

import logging
import os

import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import print_summary, set_trainable

from queens.models.surrogate_models.surrogate_model import SurrogateModel
from queens.utils.gpf_utils import extract_block_diag, init_scaler, set_transform_function

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
_logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


class GPflowSVGPModel(SurrogateModel):
    """Class for creating SVGP regression model based on GPFlow.

    **Key reference:**
        J. Hensman, A. Matthews, and Z. Ghahramani, “Scalable Variational Gaussian Process
        Classification” in Artificial Intelligence and Statistics, Feb. 2015, pp. 351–360.
        https://proceedings.mlr.press/v38/hensman15.html

    Attributes:
        number_posterior_samples (int): Number of posterior samples.
        mini_batch_size (int): Minibatch size to speed up computation of ELBO.
        number_training_iterations (int): Number of iterations in optimizer for training.
        training_data (list): List of training datasets.
        number_input_dimensions (int): Dimensionality of random features/input dimension.
        model (list): List of GPFlow based stochastic variational GP (SVGP).
        scaler_x (sklearn scaler object): Scaler for inputs.
        scaler_y (sklearn scaler object): Scaler for outputs.
        dimension_output (int): Dimensionality of the output (quantities of interest).
        seed (int): random seed
        num_inducing_points (int): Number of inducing points
        dimension_lengthscales (int): Dimension of lengthscales
        train_inducing_points_location (bool): if true, location of inducing points is trained
        train_likelihood_variance (bool): if true, likelihood variance is trained
    """

    def __init__(
        self,
        training_iterator=None,
        testing_iterator=None,
        eval_fit=None,
        error_measures=None,
        plotting_options=None,
        number_posterior_samples=None,
        mini_batch_size=100,
        number_training_iterations=10000,
        seed=41,
        number_inducing_points=100,
        dimension_lengthscales=None,
        train_inducing_points_location=False,
        train_likelihood_variance=True,
    ):
        """Initialize an instance of the GPFlow SVGP model.

        Args:
            training_iterator (Iterator): Iterator to evaluate the subordinate model with the
                                          purpose of getting training data
            testing_iterator (Iterator): Iterator to evaluate the subordinate model with the purpose
                                         of getting testing data
            eval_fit (str): How to evaluate goodness of fit
            error_measures (list): List of error measures to compute
            plotting_options (dict): plotting options
            number_posterior_samples (int): number of posterior samples
            mini_batch_size (int): minibatch size to speed up computation of ELBO
            number_training_iterations (int): number of iterations in optimizer for training
            seed (int): random seed
            number_inducing_points (int): Number of inducing points
            dimension_lengthscales (int): Dimension of lengthscales
            train_inducing_points_location (bool): if true, location of inducing points is trained
            train_likelihood_variance (bool): if true, likelihood variance is trained
        """
        super().__init__(
            training_iterator=training_iterator,
            testing_iterator=testing_iterator,
            eval_fit=eval_fit,
            error_measures=error_measures,
            plotting_options=plotting_options,
        )
        self.number_posterior_samples = number_posterior_samples
        self.mini_batch_size = mini_batch_size
        self.number_training_iterations = number_training_iterations
        self.training_data = None
        self.number_input_dimensions = None
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        self.dimension_output = None
        self.seed = seed
        self.num_inducing_points = number_inducing_points
        self.dimension_lengthscales = dimension_lengthscales
        self.train_inducing_points_location = train_inducing_points_location
        self.train_likelihood_variance = train_likelihood_variance

    def setup(self, x_train, y_train):
        """Setup surrogate model.

        Args:
            x_train (np.array): training inputs
            y_train (np.array): training outputs
        """
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self._init_training_dataset(x_train, y_train)
        self._build_model()

    def train(self):
        """Train the GP."""
        optimizer = tf.optimizers.Adam()
        for i in range(self.dimension_output):
            training_iterations = iter(self.training_data[i].batch(self.mini_batch_size))
            training_loss = self.model[i].training_loss_closure(training_iterations, compile=True)

            @tf.function
            def optimization_step(training_loss=training_loss, model=self.model[i]):
                optimizer.minimize(training_loss, model.trainable_variables)

            for step in range(self.number_training_iterations):
                optimization_step()
                if (step + 1) % 100 == 0:
                    _logger.info(
                        'Iter: %d/%d, Loss: %.2e',
                        step + 1,
                        self.number_training_iterations,
                        training_loss().numpy(),
                    )

            print_summary(self.model[i])

    def predict(self, x_test, support='f', full_cov=False):
        """Predict the posterior distribution at *x_test*.

        Options:
            #. 'f': predict the latent function values
            #. 'y': predict values of the new observations (including noise)

        Args:
            x_test (np.ndarray): New inputs where to make predictions
            support (str): Probabilistic support of random process (default: 'y')
            full_cov (bool): Boolean that specifies whether the entire posterior covariance
                             matrix should be returned or only the posterior variance

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
            posterior samples at *x_test*
        """
        assert support in ['f', 'y'], "Unknown input for support."
        x_test = np.atleast_2d(x_test).reshape((-1, self.number_input_dimensions))
        number_test_samples = x_test.shape[0]
        x_test = self.scaler_x.transform(x_test)

        output = {
            'result': [],
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

            output['result'].append(mean)
            if support == 'f' and full_cov is True:
                output['variance'].append(np.squeeze(var, axis=0))
                output['variance_diagonal'].append(
                    extract_block_diag(np.squeeze(var, axis=0), output['result'][-1].shape[1])
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

    def _init_training_dataset(self, x_train, y_train):
        """Initialize the training data set.

        Args:
            x_train (np.array): training inputs
            y_train (np.array): training outputs
        """
        if len(x_train.shape) == 1:
            self.number_input_dimensions = 1
        else:
            self.number_input_dimensions = x_train.shape[1]

        if len(y_train.shape) != 3:
            y_train = np.expand_dims(y_train, axis=1)
        self.dimension_output = y_train.shape[1]
        y_train = np.moveaxis(y_train, 1, 2)
        y_train = y_train.reshape(-1, self.dimension_output)

        self.scaler_x, self.x_train = init_scaler(x_train)
        self.scaler_y, self.y_train = init_scaler(y_train)

        self.training_data = []
        for i in range(self.dimension_output):
            self.training_data.append(
                tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train[:, [i]]))
                .repeat()
                .shuffle(x_train.shape[0], seed=self.seed)
            )

    def _build_model(self):
        """Build the SVGP model."""
        lengthscales_0 = 0.1 * np.ones(self.dimension_lengthscales)
        variances_0 = max(abs(np.max(self.y_train) - np.min(self.y_train)), 1e-6)

        inducing_points = self._init_inducing_points()

        self.model = []
        kernel = []
        for i in range(self.dimension_output):
            kernel.append(gpf.kernels.RBF(lengthscales=lengthscales_0, variance=variances_0))

            self.model.append(
                gpf.models.SVGP(
                    kernel[i],
                    gpf.likelihoods.Gaussian(),
                    inducing_points,
                    num_data=self.x_train.shape[0],
                )
            )

            if not self.train_inducing_points_location:
                gpf.set_trainable(self.model[i].inducing_variable, False)

            if not self.train_likelihood_variance:
                self.model[i].likelihood.variance.assign(
                    1.1e-6
                )  # small value for numerical stability
                set_trainable(self.model[i].likelihood.variance, False)

            self.model[i].kernel.lengthscales = set_transform_function(
                self.model[i].kernel.lengthscales,
                tfp.bijectors.Exp(),
            )
            self.model[i].kernel.variance = set_transform_function(
                self.model[i].kernel.variance, tfp.bijectors.Exp()
            )

    def _init_inducing_points(self):
        """Initialize inducing points.

        Returns:
            inducing_points (np.ndarray): inducing points
        """
        number_inducing_points = min(self.x_train.shape[0], self.num_inducing_points)
        idx = np.arange(0, self.x_train.shape[0], 1)
        idx = np.random.choice(idx, size=number_inducing_points, replace=False)
        inducing_points = self.x_train[idx, :].copy()

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

        for current_key in ['result', 'variance', 'variance_diagonal', 'post_samples']:
            _squeeze_array(current_key)

        return output
