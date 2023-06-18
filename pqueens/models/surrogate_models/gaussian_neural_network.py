"""Gaussian Neural Network regression model."""

import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from pqueens.regression_approximations.regression_approximation import RegressionApproximation
from pqueens.utils.random_process_scaler import Scaler
from pqueens.utils.valid_options_utils import get_option
from pqueens.visualization.gaussian_neural_network_vis import plot_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tfd = tfp.distributions
Dense = tf.keras.layers.Dense
tf.keras.backend.set_floatx('float64')
_logger = logging.getLogger(__name__)

# Use GPU acceleration if possible
if tf.test.gpu_device_name() != '/device:GPU:0':
    _logger.debug('WARNING: GPU device not found.')
else:
    _logger.debug('SUCCESS: Found GPU: %s', tf.test.gpu_device_name())


class GaussianNeuralNetwork(RegressionApproximation):
    """Class for creating a neural network that parameterizes a Gaussian.

    The network can handle heteroskedastic noise and an arbitrary nonlinear functions.

    Attributes:
        x_train (np.array): Training inputs
        y_train (np.array): Training outputs
        nn_model (tf.model):  Tensorflow based Bayesian neural network model
        num_epochs (int): Number of training epochs for variational optimization
        optimizer_seed (int): Random seed used for initialization of stochastic gradient decent
                              optimizer
        verbosity_on (bool): Boolean for model verbosity during training. True=verbose
        batch_size (int): Size of data-batch (smaller than the training data size)
        scaler_x (obj): Scaler for inputs
        scaler_y (obj): Scaler for outputs
        loss_plot_path (str): Path to determine whether loss plot should be produced
                              (yes if provided). Plot will be saved at path location.
        refinement_learning_rate_decay (float): Decrease of epochs in refinements
        mean_function (function): Mean function of the Gaussian Neural Network
        gradient_mean_function (function): Gradient of the mean function of the Gaussian
                                           Neural Network
    """

    def __init__(
        self,
        x_train,
        y_train,
        nn_model,
        num_epochs,
        optimizer_seed,
        verbosity_on,
        batch_size,
        scaler_x,
        scaler_y,
        loss_plot_path,
        refinement_epochs_decay,
        mean_function,
        gradient_mean_function,
    ):
        """Initialize an instance of the Gaussian Bayesian Neural Network.

        Args:
            x_train (np.array): Training inputs
            y_train (np.array): Training outputs
            num_posterior_samples (int): Number of posterior sample functions
            nn_model (obj): Tensorflow probability model instance
            num_epochs (int): Number of epochs used for variational training of the BNN
            optimizer_seed (int): Random seed for stochastic optimization routine
            verbosity_on (bool): Boolean for model verbosity during training. True=verbose
            batch_size (int): Size of data-batch (smaller than the training data size)
            loss_plot_path (str): Path to determine whether loss plot should be produced
                                  (yes if provided). Plot will be saved at path location.
            scaler_x (obj): Data scaling object for input data
            scaler_y (obj): Data scaling object for output data
            refinement_learning_rate_decay (float): Decrease of epochs in refinements
            mean_function (function): Mean function of the Gaussian Neural Network
            gradient_mean_function (function): Gradient of the mean function of the Gaussian
                                               Neural Network

        Returns:
            Instance of GaussianBayesianNeuralNetwork
        """
        self.x_train = x_train
        self.y_train = y_train
        self.nn_model = nn_model
        self.num_epochs = num_epochs
        self.optimizer_seed = optimizer_seed
        self.verbosity_on = verbosity_on
        self.batch_size = batch_size
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.loss_plot_path = loss_plot_path
        self.num_refinements = 0
        self.refinement_epochs_decay = refinement_epochs_decay
        self.mean_function = mean_function
        self.gradient_mean_function = gradient_mean_function

    @classmethod
    def from_config_create(cls, config, approx_name, x_train, y_train):
        """Create approximation from options dictionary.

        Args:
            config (dict): Dictionary with problem description (input file)
            x_train (np.array):    Training inputs
            y_train (np.array):    Training outputs
            approx_name (str):     Name of the approximation options in input file

        Returns:
            Tensorflow Bayesian neural network object
        """
        approx_options = config[approx_name]
        num_epochs = approx_options.get('num_epochs')
        batch_size = approx_options.get('batch_size')
        adams_training_rate = approx_options.get('adams_training_rate')
        optimizer_seed = approx_options.get('optimizer_seed')
        verbosity_on = approx_options.get('verbosity_on')
        nodes_per_hidden_layer = approx_options.get('nodes_per_hidden_layer_lst')
        activation_per_hidden_layer = approx_options.get('activation_per_hidden_layer_lst')
        kernel_initializer = approx_options.get('kernel_initializer')
        nugget_std = approx_options.get('nugget_std')
        loss_plot_path = approx_options.get("loss_plot_path", False)
        refinement_epochs_decay = approx_options.get("refinement_epochs_decay", 0.75)

        # get normalization bool
        scaler_settings = config[approx_name].get("data_scaling")
        scaler_obj_x = Scaler.from_config_create_scaler(scaler_settings)
        scaler_obj_y = Scaler.from_config_create_scaler(scaler_settings)

        # check mean function and subtract from y_train
        valid_mean_function_types = {
            "zero": (lambda x: 0, lambda x: 0),
            "identity_multi_fidelity": (
                lambda x: np.atleast_2d(x[:, 0]).T,
                lambda x: np.hstack((np.ones(x.shape[0]).reshape(-1, 1), np.zeros(x[:, 1:].shape))),
            ),
        }

        mean_function_type = approx_options.get("mean_function_type", "zero")
        mean_function, gradient_mean_function = get_option(
            valid_mean_function_types, mean_function_type, "mean_function_type"
        )
        y_train = y_train - mean_function(x_train)

        scaler_obj_x.fit(x_train)
        x_train = scaler_obj_x.transform(x_train)
        scaler_obj_y.fit(y_train)
        y_train = scaler_obj_y.transform(y_train)

        nn_model = cls._build_model(
            adams_training_rate,
            nodes_per_hidden_layer,
            activation_per_hidden_layer,
            y_train,
            kernel_initializer,
            nugget_std,
        )

        return cls(
            x_train,
            y_train,
            nn_model,
            num_epochs,
            optimizer_seed,
            verbosity_on,
            batch_size,
            scaler_obj_x,
            scaler_obj_y,
            loss_plot_path,
            refinement_epochs_decay,
            mean_function,
            gradient_mean_function,
        )

    @classmethod
    def _build_model(
        cls,
        adams_training_rate,
        nodes_per_hidden_layer,
        activation_per_hidden_layer,
        y_train,
        kernel_initializer,
        nugget_std,
    ):
        """Build/compile the neural network.

        We use a regular densely connected
        NN, which is parameterizing mean and variance of a Gaussian
        distribution. The network can be arbitrary deep and wide and can use
        different (nonlinear) activation functions.

        Args:
            adams_training_rate (float): Training rate for the ADAMS gradient decent optimizer
            nodes_per_hidden_layer (lst): List containing number of nodes per hidden layer of
                                          the Neural Network. The length of the list
                                          defines the deepness of the model and the values the
                                          width of the individual layers.
            activation_per_hidden_layer (list): List with strings encoding the activation
                                                function that shall be used for the
                                                respective hidden layer of the  Neural
                                                Network
            y_train (np.array): training output array
            kernel_initializer (str): Type of kernel initialization for neural network
            nugget_std (float): Nugget standard deviation for robustness

        Returns:
            model (obj): Tensorflow probability model instance
        """
        # hidden layers
        output_dim = y_train.shape[1]

        dense_architecture = [
            Dense(
                int(num_nodes),
                activation=activation,
                kernel_initializer=kernel_initializer,
            )
            for num_nodes, activation in zip(nodes_per_hidden_layer, activation_per_hidden_layer)
        ]

        # Gaussian output layer
        output_layer = [
            Dense(
                2 * output_dim,
                activation='linear',
            ),
            tfp.layers.DistributionLambda(
                lambda d: tfd.Normal(
                    loc=d[..., :output_dim],
                    scale=nugget_std + tf.math.softplus(0.1 * d[..., output_dim:]),
                )
            ),
        ]
        dense_architecture.extend(output_layer)
        model = tf.keras.Sequential(dense_architecture)

        # compile the Tensorflow model
        optimizer = tf.optimizers.Adamax(learning_rate=adams_training_rate)

        model.compile(
            optimizer=optimizer,
            loss=cls.negative_log_likelihood,
        )

        return model

    @staticmethod
    def negative_log_likelihood(y, rv_y):
        """Negative log. likelihood of (tensorflow) random variable rv_y.

        Args:
            y (float): Value/Realization of the random variable
            rv_y (obj): Tensorflow probability random variable object

        Returns:
            negative_log_likelihood (float): Negative logarithmic likelihood of rv_y at y
        """
        negative_log_likelihood = -rv_y.log_prob(y)
        return negative_log_likelihood

    def update_training_data(self, x_train, y_train):
        """Update the training data of the model.

        Args:
            x_train (np.array): Training input array
            y_train (np.array): Training output array
        """
        num_old_samples = self.x_train.shape[0]
        x_train_new = self.scaler_x.transform(x_train[num_old_samples:].T).T
        y_train_new = self.scaler_y.transform(y_train[num_old_samples:])
        self.x_train = np.vstack((self.x_train, x_train_new))
        self.y_train = np.vstack((self.y_train, y_train_new))

    def train(self):
        """Train the Bayesian neural network.

        We ues the previous defined optimizers in the model build and configuration.
        We allow tensorflow's early stopping here to stop the optimization routine when the loss-
        function starts to increase again over several iterations.

        Returns:
            None
        """
        # make epochs adaptive with a simple schedule, lower bound is 1/5 of the initial epoch
        if self.num_refinements > 0:
            self.num_epochs = int(
                max(self.num_epochs * self.refinement_epochs_decay, self.num_epochs / 5)
            )
        self.num_refinements += 1

        # set the random seeds for optimization/training
        tf.keras.utils.set_random_seed(self.optimizer_seed)
        history = self.nn_model.fit(
            self.x_train,
            self.y_train,
            epochs=self.num_epochs,
            verbose=self.verbosity_on,
            batch_size=self.batch_size,
        )

        # print out the model summary
        self.nn_model.summary()

        if self.loss_plot_path:
            plot_loss(history, self.loss_plot_path)

    def predict(self, x_test, support='y', gradient_bool=False):
        """Predict the output distribution at x_test.

        Args:
            x_test (np.array): Testing input vector for which the posterior distribution,
                               respectively point estimates should be predicted
            support (str, optional): String to define the support of the output distribution
                                    - 'y': Conditional distribution is defined on the output space
                                    - 'f': Conditional distribution is defined on the latent space
            gradient_bool (bool, optional): Boolean to configure whether gradients should be
                                            returned as well

        Returns:
            output (dict): Dictionary with posterior output statistics
        """
        if support == 'f':
            raise NotImplementedError('Support "f" is not implemented yet.')

        if gradient_bool:
            output = self.predict_and_gradient(x_test)
        else:
            output = self.predict_y(x_test)

        output["x_test"] = x_test
        return output

    def predict_y(self, x_test):
        """Predict the posterior mean and variance.

        Prediction is conducted w.r.t. to the output space "y".

        Args:
            x_test (np.array): Testing input vector for which the posterior distribution,
                               respectively point estimates should be predicted

        Returns:
            output (dict): Dictionary with posterior output statistics
        """
        x_test_transformed = self.scaler_x.transform(x_test)
        yhat = self.nn_model(x_test_transformed)
        mean_pred = np.atleast_2d(yhat.mean()).T
        var_pred = np.atleast_2d(yhat.variance()).T

        output = {"variance_untransformed": var_pred}
        output["variance"] = (self.scaler_y.inverse_transform_std(np.sqrt(var_pred)) ** 2).reshape(
            -1, 1
        )
        output["mean"] = self.scaler_y.inverse_transform_mean(mean_pred).reshape(
            -1, 1
        ) + self.mean_function(x_test)

        return output

    def predict_and_gradient(self, x_test):
        """Predict the mean, variance and their gradients at x_test.

        Args:
            x_test (np.array): Testing input vector for which the posterior
                               distribution, respectively point estimates should be
                               predicted

        Returns:
            output (dict): Dictionary with posterior output statistics
        """
        x_test_transformed = self.scaler_x.transform(x_test)
        x_test_tensorflow = tf.Variable(x_test_transformed)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_test_tensorflow)
            yhat = self.nn_model(x_test_tensorflow)
            mean_pred = yhat.mean()
            var_pred = yhat.variance()

        grad_mean = tape.gradient(mean_pred, x_test_tensorflow).numpy()
        grad_var = tape.gradient(var_pred, x_test_tensorflow).numpy()

        mean_pred = np.array(mean_pred.numpy()).reshape(-1, 1)
        var_pred_untransformed = np.array(var_pred.numpy()).reshape(-1, 1)

        # write mean and variance to output dictionary
        output = {
            "mean": self.scaler_y.inverse_transform_mean(mean_pred).reshape(-1, 1)
            + self.mean_function(x_test)
        }
        output["variance"] = (
            self.scaler_y.inverse_transform_std(np.sqrt(var_pred_untransformed)) ** 2
        ).reshape(-1, 1)

        # write gradients to output dictionary
        output["grad_mean"] = self.scaler_y.inverse_transform_grad_mean(
            grad_mean, self.scaler_x.standard_deviation
        ) + self.gradient_mean_function(x_test)

        output["grad_var"] = self.scaler_y.inverse_transform_grad_var(
            grad_var,
            var_pred_untransformed,
            output["variance"],
            self.scaler_x.standard_deviation,
        )

        return output
