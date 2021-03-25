import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp

from pqueens.regression_approximations.regression_approximation import RegressionApproximation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tfd = tfp.distributions
DenseVar = tfp.layers.DenseVariational
tf.keras.backend.set_floatx('float64')

# Use GPU acceleration if possible
if tf.test.gpu_device_name() != '/device:GPU:0':
    print('WARNING: GPU device not found.')
else:
    print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))


class GaussianBayesianNeuralNetwork(RegressionApproximation):
    """
    Class for creating a Bayesian neural network with Gaussian conditional distribution based on
    Tensorflow Probability.

    The network can handle heteroskedastic noise and an arbitrary nonlinear functional. As we use
    Tensorflow variational layers and learn the mean and variance function of a Gaussian
    distribution, the network is able to handle epistemic and aleatory uncertainty.

    Attributes:
        x_train (np.array): Training inputs
        y_train (np.array): Training outputs
        bnn_model (tf.model):  Tensorflow based Bayesian neural network model
        num_posterior_samples (int): Number of posterior sample functions (realizations of
                                     Bayesian neural network)
        num_samples_statistics (int): Number of samples to approximate posterior statistics
        num_epochs (int): Number of training epochs for variational optimization
        optimizer_seed (int): Random seed used for initialization of stochastic gradient decent
                              optimizer
        verbosity_on (bool): Boolean for model verbosity during training. True=verbose
        model_realizations_lst (lst): List with different neural network realizations
                                      (epistemic uncertainty)

    """

    def __init__(
        self,
        x_train,
        y_train,
        num_posterior_samples,
        num_samples_statistics,
        bnn_model,
        num_epochs,
        optimizer_seed,
        verbosity_on,
    ):
        """
        Initialize an instance of the Gaussian Bayesian Neural Network

        Args:
            x_train (np.array): Training inputs
            y_train (np.array): Training outputs
            num_posterior_samples (int): Number of posterior sample functions
            num_samples_statistics (int): Number of samples to approximate posterior statistics
            bnn_model (obj): Tensorflow probability model instance
            num_epochs (int): Number of epochs used for variational training of the BNN
            optimizer_seed (int): Random seed for stochastic optimization routine
            verbosity_on (bool): Boolean for model verbosity during training. True=verbose



        Returns:
            Instance of GaussianBayesianNeuralNetwork

        """
        self.x_train = x_train
        self.y_train = y_train
        self.num_posterior_samples = num_posterior_samples
        self.num_samples_statistics = num_samples_statistics
        self.bnn_model = bnn_model
        self.num_epochs = num_epochs
        self.optimizer_seed = optimizer_seed
        self.verbosity_on = verbosity_on
        self.model_realizations_lst = None

    @classmethod
    def from_config_create(cls, config, approx_name, x_train, y_train):
        """
        Create approximation from options dictionary

        Args:
            config (dict): Dictionary with problem description (input file)
            x_train (np.array):    Training inputs
            y_train (np.array):    Training outputs
            approx_name (str):     Name of the approximation options in input file

        Returns:
            Tensorflow Bayesian neural network object

        """
        approx_options = config[approx_name]
        num_posterior_samples = approx_options.get('num_posterior_samples', None)
        num_samples_statistics = approx_options.get('num_samples_statistics', None)
        num_epochs = approx_options.get('num_epochs')
        adams_training_rate = approx_options.get('adams_training_rate')
        optimizer_seed = approx_options.get('optimizer_seed')
        verbosity_on = approx_options.get('verbosity_on')

        nodes_per_hidden_layer_lst = approx_options.get('nodes_per_hidden_layer_lst')
        activation_per_hidden_layer_lst = approx_options.get('activation_per_hidden_layer_lst')

        bnn_model = cls._build_model(
            x_train,
            adams_training_rate,
            nodes_per_hidden_layer_lst,
            activation_per_hidden_layer_lst,
        )

        return cls(
            x_train,
            y_train,
            num_posterior_samples,
            num_samples_statistics,
            bnn_model,
            num_epochs,
            optimizer_seed,
            verbosity_on,
        )

    @classmethod
    def _build_model(
        cls,
        x_train,
        adams_training_rate,
        nodes_per_hidden_layer_lst,
        activation_per_hidden_layer_lst,
    ):
        """
        Build/compile the Bayesian neural network.
        We use a regular densely connected NN, which is parameterizing mean and variance of a
        Gaussian distribution. The network can be arbitrary deep and wide and can use different (
        nonlinear) activation functions.


        Args:
            x_train (np.array): Input training points for the Bayesian Neural Network
            adams_training_rate (float): Training rate for the ADAMS gradient decent optimizer
            nodes_per_hidden_layer_lst (lst): List containing number of nodes per hidden layer of
                                              the Bayesian Neural Network. The length of the list
                                              defines the deepness of the model and the values the
                                              width of the individual layers.
            activation_per_hidden_layer_lst (lst): List with strings encoding the activation
                                                   function that shall be used for the
                                                   respective hidden layer of the Bayesian Neural
                                                   Network

        Returns:
            model (obj): Tensorflow probability model instance

        """
        dense_architecture = [
            DenseVar(
                int(num_nodes),
                cls.mean_field_variational_distribution,
                cls.prior_trainable,
                activation=activation,
                kl_weight=1 / x_train.shape[0],
            )
            for num_nodes, activation in zip(
                nodes_per_hidden_layer_lst, activation_per_hidden_layer_lst
            )
        ]
        output_layer = [
            DenseVar(
                2,
                cls.mean_field_variational_distribution,
                cls.prior_trainable,
                kl_weight=1 / x_train.shape[0],
            ),
            tfp.layers.DistributionLambda(
                lambda d: tfd.Normal(
                    loc=d[..., :1], scale=1e-3 + tf.math.softplus(0.01 * d[..., 1:])
                )
            ),
        ]
        dense_architecture.extend(output_layer)
        model = tf.keras.Sequential(dense_architecture)

        # compile the Tensorflow model
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=adams_training_rate),
            loss=cls.negative_log_likelihood,
        )
        return model

    @staticmethod
    def negative_log_likelihood(y, rv_y):
        """
        Negative logarithmic likelihood of (tensorflow) random variable rv_y evaluated at location y

        Args:
            y (float): Value/Realization of the random variable
            rv_y (obj): Tensorflow probability random variable object

        Returns:
            negloglik (float): Negative logarithmic likelihood of rv_y at y

        """
        negloglik = -rv_y.log_prob(y)
        return negloglik

    @staticmethod
    def prior_trainable(kernel_size, bias_size=0, dtype=None):
        """
        Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
        Note this is a special hybrid case of prior which is actually trainable see "Empirical
        Bayes" for more background on this topic.

        Args:
            kernel_size (int): Number of weight parameters in the NN
            bias_size (int): Number of bias parameters in the NN
            dtype (str):  DataType string

        Returns:
            prior (obj): Tensorflow probability Gaussian prior distribution over biases and
                         weights of the Bayesian NN, taking the mean of the Gaussian as an input
                         variable for the prior distribution and fixing the variance

        """
        # TODO check this prior and make it full Bayesian?
        n = kernel_size + bias_size

        # note: the mean of the Gaussian is an input variable here, hence we have n*mean input
        # variables; one mean per node of the NN
        prior_distribution = tf.keras.Sequential(
            [
                tfp.layers.VariableLayer(n, dtype=dtype),
                tfp.layers.DistributionLambda(
                    lambda mean: tfd.Independent(
                        tfd.Normal(loc=mean, scale=1), reinterpreted_batch_ndims=1
                    )
                ),
            ]
        )
        return prior_distribution

    @staticmethod
    def mean_field_variational_distribution(kernel_size, bias_size=0, dtype=None):
        """
        Variational distribution that approximates the true posterior. Here we use a Gaussian
        mean field approach, such that every parameter in the NN is approximated with a Gaussian
        distribution, parameterized by a mean and variance.

        Args:
            kernel_size (int): Number of weight parameters in the NN
            bias_size (int): Number of bias parameters in the NN
            dtype (str):  DataType string

        Returns:
            mean_field_variational_distr (obj): Tensorflow Probability mean field variational
            distribution for weights and biases; Each node (with weight and bias) has two Gaussians
            that are individually optimized via their mean and variance.
            Hence the parameterization of the variational distribution has n * 2 parameters with
            n being the number of nodes in the NN.

        """
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.0))
        mean_field_variational_distr = tf.keras.Sequential(
            [
                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                tfp.layers.DistributionLambda(
                    lambda t: tfd.Independent(
                        tfd.Normal(loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                        reinterpreted_batch_ndims=1,
                    )
                ),
            ]
        )
        return mean_field_variational_distr

    def train(self):
        """
        Train the Bayesian neural network using the previous defined optimizers in the model
        build and configuration. We allow tensorflow's early stopping here to stop the
        optimization routine when the loss-function starts to increase again over several
        iterations.

        Returns:
            None

        """
        # set the random seeds for optimization/training
        np.random.seed(self.optimizer_seed)
        tf.random.set_seed(self.optimizer_seed)

        self.bnn_model.fit(
            self.x_train,
            self.y_train,
            epochs=self.num_epochs,
            verbose=self.verbosity_on,
        )

        # print out the model summary
        self.bnn_model.summary()

    def predict(self, x_test, support='y', full_cov=False):
        """

        Args:
            x_test (np.array): Testing input vector for which the posterior distribution,
                               respectively point estimates should be predicted
            support (str): String to determine which output
            full_cov (bool): Boolean flag for prediction posterior covariance matrix (True) or
                             posterior variance (False) at x_test

        Returns:
            output (dict): Dictionary with posterior output statistics

        """
        if self.model_realizations_lst is None:
            self.model_realizations_lst = [
                self.bnn_model for _ in range(self.num_samples_statistics)
            ]

        if support == 'y':
            predict_method = self.predict_y
        elif support == 'f':
            predict_method = self.predict_f
        else:
            raise RuntimeError('No suitable prediction method could be selected. Abort...')

        output = predict_method(x_test, full_cov=full_cov)

        return output

    def predict_y(self, x_test, full_cov=False):
        """
        Predict the posterior mean, variance or covariance of the posterior distribution at
        x_test (wrt to y) that combines epistemic and aleatory uncertainty.

        Args:
            x_test (np.array): Testing input vector for which the posterior distribution,
                               respectively point estimates should be predicted
            full_cov (bool): Boolean flag for prediction posterior covariance matrix (True) or
                             posterior variance (False) at x_test

        Returns:
            output (dict): Dictionary with posterior output statistics

        """
        output = {}

        # sample over possible models (epistemic uncertainty)
        y_rv_models = [
            model_realization(x_test) for model_realization in self.model_realizations_lst
        ]
        # get mean of the individual gaussian models (aleatory uncertainty)
        y_rv_model_means = np.array([y_rv.mean() for y_rv in y_rv_models]).squeeze()
        # combine both sources of uncertainty averaging them
        output["mean"] = np.atleast_2d(y_rv_model_means.mean(axis=0)).T

        # repeat the former process for variance/covariance estimates
        if full_cov is False:
            y_rv_model_var = np.array([y_rv.variance() for y_rv in y_rv_models])
            output["variance"] = y_rv_model_var.mean(axis=0)
        elif full_cov is True:
            y_rv_model_var = (
                (np.array([y_rv.variance() for y_rv in y_rv_models])).squeeze().mean(axis=0)
            )
            y_rv_model_cov = np.cov(y_rv_model_means.T) + np.diag(y_rv_model_var)
            output["variance"] = y_rv_model_cov

        return output

    def predict_f(self, x_test, full_cov=False):
        """
        Predict the posterior mean, variance or covariance of the posterior distribution wrt to
        the latent function f at test points x_test. This prediction only accounts for the
        epistemic uncertainty and neglects the aleatory uncertainty.

        Args:
            x_test (np.array): Testing input vector for which the posterior distribution,
                               respectively point estimates should be predicted
            full_cov (bool): Boolean flag for prediction posterior covariance matrix (True) or
                             posterior variance (False) at x_test

        Returns:
            output (dict): Dictionary with posterior output statistics

        """
        output = {}

        # sample over possible models (epistemic uncertainty)
        y_rv_models = [
            model_realization(x_test) for model_realization in self.model_realizations_lst
        ]
        y_rv_model_means = np.array([y_rv.mean() for y_rv in y_rv_models])
        output["mean"] = y_rv_model_means.mean(axis=0)

        # repeat the former process for variance/covariance estimates
        if full_cov is False:
            output["variance"] = y_rv_model_means.var(axis=0)
        elif full_cov is True:
            samples = self.predict_f_samples(x_test, self.num_posterior_samples)
            output["variance"] = np.cov(samples)

        return output

    def predict_f_samples(self, x_test, num_samples):
        """
        Sample the latent function (without noise) that is in this this case realizations of the
        mean predictions of the Bayesian neural network which are themselves uncertain.

        Args:
            x_test (np.array): Input testing point at which the samples should be realized
            num_samples (int): Number of posterior samples of the latent function f

        Returns:
            samples (np.array): Samples of the latent function at locations x_test

        """
        y_rv_models = [
            model_realization(x_test) for model_realization in self.model_realizations_lst
        ]
        y_rv_model_mean_samples = np.array([y_rv.mean() for y_rv in y_rv_models])
        samples = np.reshape(y_rv_model_mean_samples, (x_test.shape[0], num_samples))

        return samples

    def predict_y_samples(self, x_test, num_samples):
        """
        Sampling from the posterior (wrt to y) that combines epistemic and aleatory uncertainty.
        This will generally lead to noisy samples in contrast to smooth samples from the latent
        function as noise assumption is conditionally independent for each point x_test.

        Args:
            x_test (np.array): Testing input locations for the posterior distribution
            num_samples (int): Number of posterior samples at x_test

        Returns:
            samples (np.array): Posterior samples wrt y

        """
        # sample the different model realizations (epistemic uncertainty)
        y_rv_models = [
            model_realization(x_test) for model_realization in self.model_realizations_lst
        ]
        # from each model draw one sample
        y_rv_model_samples = np.array(
            [
                y_rv.sample(int(self.num_posterior_samples / self.num_samples_statistics))
                for y_rv in y_rv_models
            ]
        )

        samples = np.reshape(y_rv_model_samples, (x_test.shape[0], num_samples))
        return samples
