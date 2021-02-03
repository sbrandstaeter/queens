from .regression_approximation import RegressionApproximation
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gpf
from sklearn.cluster import KMeans
from gpflow.utilities import print_summary


class HeteroskedasticGP(RegressionApproximation):
    """ Class for creating heteroskedastic GP based regression model based on GPFlow

    This class constructs a GP regression, currently using a GPFlow model.
    Currently, a lot of parameters are still hard coded, which will be
    improved in the future.

    The basic idea of this latent variable GP model can be found in [1-3]

    Attributes:
        x_train (np.array): Training inputs
        y_train (np.array): Training outputs
        gpf_model (gpf.model):  gpflow based heteroskedastic Gaussian process model
        num_posterior_samples (int): Number of posterior GP samples (realizations of posterior GP)
        num_inducing_points (int): Number of inducing points for variational GPs
        optimizer (obj): Tensorflow optimization object
        num_epochs (int): Number of training epochs for variational optimization
        optimizer_seed (int): Random seed used for initialization of stochastic gradient decent
                              optimizer

    References:
        [1]: https://gpflow.readthedocs.io/en/develop/notebooks/advanced/heteroskedastic.html
        [2]: Saul, A. D., Hensman, J., Vehtari, A., & Lawrence, N. D. (2016, May).
             Chained gaussian processes. In Artificial Intelligence and Statistics (pp. 1431-1440).
        [3]: Hensman, J., Fusi, N., & Lawrence, N. D. (2013). Gaussian processes for big data.
             arXiv preprint arXiv:1309.6835.

    """

    def __init__(
        self,
        x_train,
        y_train,
        num_posterior_samples,
        num_inducing_points,
        gpf_model,
        optimizer,
        num_epochs,
        optimizer_seed,
    ):
        """
        Initialize an instance of the Heteroskedastic GPflow class

        Args:
            x_train (np.array): Training inputs
            y_train (np.array): Training outputs
            num_posterior_samples: Number of posterior GP samples
            num_inducing_points: Number of inducing points for variational GP approximation
            gpf_model (obj): GPFlow model instance
            optimizer (obj): GPflow optimization object
            num_epochs (int): Number of epochs used for variational training of the GP
            optimizer_seed (int): Random seed for stochastic optimization routine

        Returns:
            None

        """

        self.x_train = x_train
        self.y_train = y_train
        self.num_posterior_samples = num_posterior_samples
        self.num_inducing_points = num_inducing_points
        self.gpf_model = gpf_model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.optimizer_seed = optimizer_seed

    @classmethod
    def from_config_create(cls, config, approx_name, x_train, y_train):
        """ Create approximation from options dictionary

        Args:
            config (dict): Dictionary with problem description (input file)
            x_train (np.array):    Training inputs
            y_train (np.array):    Training outputs
            approx_name (str):     Name of the approximation options in input file

        Returns:
            gpflow heteroskedastic GP object

        """
        approx_options = config[approx_name]
        num_posterior_samples = approx_options.get('num_posterior_samples', None)
        num_inducing_points = approx_options.get('num_inducing_points')
        num_epochs = approx_options.get('num_epochs')
        adams_training_rate = approx_options.get('adams_training_rate')
        optimizer_seed = approx_options.get('optimizer_seed')

        gpf_model = HeteroskedasticGP._build_model(num_inducing_points, x_train)
        optimizer = HeteroskedasticGP._build_optimizer(
            x_train, y_train, gpf_model, adams_training_rate
        )

        return cls(
            x_train,
            y_train,
            num_posterior_samples,
            num_inducing_points,
            gpf_model,
            optimizer,
            num_epochs,
            optimizer_seed,
        )

    def train(self):
        """ Train the variational by minimizing the variational loss in variational EM step"""
        np.random.seed(self.optimizer_seed)
        for epoch in range(1, self.num_epochs + 1):
            self.optimizer()
            if epoch % 20 == 0 and epoch > 0:
                data = (self.x_train, self.y_train)
                loss_fun = self.gpf_model.training_loss_closure(data)
                print(
                    f'Progress: {epoch / self.num_epochs * 100 : .2f} %, Epoch {epoch}, '
                    f'Loss: {loss_fun().numpy() : .2f}'
                )

    def predict(self, x_test, full_cov=False):
        """
        Predict the posterior distribution at Xnew with respect to the data 'y'.

        Args:
            x_test (np.array): Inputs at which to evaluate or test the latent function f
            full_cov (bool): Boolean that specifies whether the entire posterior covariance matrix
                             should be returned or only the posterior variance

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
                           posterior samples at Xnew
        """
        output = self.predict_y(x_test, full_cov=full_cov)

        return output

    def predict_y(self, x_test, full_cov=False):
        """
        Compute the posterior distribution at x_test with respect to the data 'y'

        Args:
            x_test (np.array): Inputs at which to evaluate latent function f
            full_cov (bool): Boolean to decide if we want posterior variance (full_cov=False) or
                             the full posterior covariance matrix (full_cov=True)

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
                           posterior samples of latent function at x_test
        """
        x_test = np.atleast_2d(x_test).reshape((-1, self.x_train.shape[1]))
        output = {}
        mean, variance = self.gpf_model.predict_y(x_test, full_cov=full_cov)
        output["mean"] = mean.numpy()
        output["variance"] = variance.numpy()

        if self.num_posterior_samples is not None:
            output["post_samples"] = self.predict_f_samples(x_test, self.num_posterior_samples)

        return output

    def predict_f_samples(self, x_test, num_samples):
        """ Produce samples from the posterior latent function Xnew

            Args:
                x_test (np.array):  Inputs at which to evaluate latent function f
                num_samples (int):  Number of posterior realizations of GP

            Returns:
                post_samples (np.array): Posterior samples of latent functions at x_test,
                                         (the latter might be a vector/matrix of points)
        """

        post_samples = self.gpf_model.predict_f_samples(x_test, num_samples).numpy()
        if post_samples.shape[1] != 1:
            raise NotImplementedError("Multi-dimensional output is not implemented! Abort...")

        return np.reshape(post_samples, (x_test.shape[0], num_samples))

    @classmethod
    def _build_model(cls, num_inducing_points, x_train):
        """
        Build the GPflow heteroskedastic GP model

        Args:
            num_inducing_points (int): Number of inducing points used for variational GP
                                       approximation
            x_train (np.array): Training input points for GP model


        Returns:
            gpf_model (obj): Instance of a GPflow heteroskedastic model

        """
        # heteroskedastic conditional likelihood
        likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
            distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
            scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
        )

        # separate independent kernels
        kernel = gpf.kernels.SeparateIndependent(
            [
                gpf.kernels.SquaredExponential(),  # This is k1, the kernel of f1
                gpf.kernels.SquaredExponential(),  # this is k2, the kernel of f2
            ]
        )

        # Initial inducing points position Z determined by clustering training data and take centers
        # of inferred clusters
        kmeans = KMeans(n_clusters=num_inducing_points)
        kmeans.fit(x_train)
        Z = kmeans.cluster_centers_

        inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
            [
                gpf.inducing_variables.InducingPoints(Z),  # This is U1 = f1(Z1)
                gpf.inducing_variables.InducingPoints(Z),  # This is U2 = f2(Z2)
            ]
        )

        # the final model
        model = gpf.models.SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            num_latent_gps=likelihood.latent_dim,
        )

        print('The GPFlow model used in this analysis is constructed as follows:')
        print_summary(model)
        return model

    @classmethod
    def _build_optimizer(cls, x_train, y_train, gpflow_model, adams_training_rate):
        """
        Build the optimization step for the variational EM step
        Args:
            x_train (np.array): Training input points
            y_train (np.array): Training output points
            gpflow_model (obj): Instance of a GPflow model

        Returns:
            optimization_step_fun (obj): Tensorflow optimization EM-step function for variational
                                         optimization of the GP

        """

        data = (x_train, y_train)
        loss_fn = gpflow_model.training_loss_closure(data)

        gpf.utilities.set_trainable(gpflow_model.q_mu, False)
        gpf.utilities.set_trainable(gpflow_model.q_sqrt, False)

        variational_vars = [(gpflow_model.q_mu, gpflow_model.q_sqrt)]
        natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

        adam_vars = gpflow_model.trainable_variables
        adam_opt = tf.optimizers.Adam(adams_training_rate)

        # here we conduct a two step optimization
        @tf.function
        def optimization_step_fun():
            """
            Two step variational Expectation-Maximization routine for latent variable GPs

            Returns:
                None

            """
            natgrad_opt.minimize(loss_fn, variational_vars)
            adam_opt.minimize(loss_fn, adam_vars)

        return optimization_step_fun