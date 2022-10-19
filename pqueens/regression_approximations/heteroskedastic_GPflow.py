import logging
import os

import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import print_summary
from sklearn.cluster import KMeans

from pqueens.regression_approximations.regression_approximation import RegressionApproximation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
_logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


class HeteroskedasticGP(RegressionApproximation):
    """Class for creating heteroskedastic GP based regression model based on
    GPFlow.

    This class constructs a GP regression, currently using a GPFlow model.
    Currently, a lot of parameters are still hard coded, which will be
    improved in the future.

    The basic idea of this latent variable GP model can be found in [1-3]

    Attributes:
        x_train (np.array): Training inputs
        y_train (np.array): Training outputs
        model (gpf.model):  gpflow based heteroskedastic Gaussian process model
        num_posterior_samples (int): Number of posterior GP samples (realizations of posterior GP)
        num_inducing_points (int): Number of inducing points for variational GPs
        optimizer (obj): Tensorflow optimization object
        num_epochs (int): Number of training epochs for variational optimization
        random_seed_seed (int): Random seed used for initialization of stochastic gradient decent
                              optimizer
        posterior_cov_mat_y (np.array): Posterior covariance matrix of heteroskedastic GP wrt
                                        y-coordinate
        num_samples_stats (int): Number of samples used to calculate empirical
                                    variance/covariance

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
        model,
        optimizer,
        num_epochs,
        random_seed_seed,
        num_samples_stats,
        posterior_cov_mat_y,
    ):
        """Initialize an instance of the Heteroskedastic GPflow class.

        Args:
            x_train (np.array): Training inputs
            y_train (np.array): Training outputs
            num_posterior_samples: Number of posterior GP samples
            num_inducing_points: Number of inducing points for variational GP approximation
            model (obj): GPFlow model instance
            optimizer (obj): GPflow optimization object
            num_epochs (int): Number of epochs used for variational training of the GP
            random_seed_seed (int): Random seed for stochastic optimization routine and samples
            num_samples_stats (int): Number of samples used to calculate empirical
                                     variance/covariance
            posterior_cov_mat_y (np.array): Posterior covariance prediction matrix of GP calculated
                                            w.r.t. the model output `y`.

        Returns:
            None
        """

        self.x_train = x_train
        self.y_train = y_train
        self.num_posterior_samples = num_posterior_samples
        self.num_inducing_points = num_inducing_points
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.random_seed = random_seed_seed
        self.posterior_cov_mat_y = posterior_cov_mat_y
        self.num_samples_stats = num_samples_stats

    @classmethod
    def from_config_create(cls, config, approx_name, x_train, y_train):
        """Create approximation from options dictionary.

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
        random_seed = approx_options.get('random_seed')

        model = HeteroskedasticGP._build_model(num_inducing_points, x_train)
        optimizer = HeteroskedasticGP._build_optimizer(
            x_train, y_train, model, adams_training_rate, random_seed
        )
        num_samples_stats = approx_options.get('num_samples_stats')
        if num_samples_stats is None or num_samples_stats < 100:
            raise RuntimeError(
                f"You configured {num_samples_stats} number of samples for the calculation "
                "of the empirical posterior statistics. This number is either too low for "
                "reliable results or not a valid input. Please provide a valide integrer input "
                "greater than 100. Abort ..."
            )

        # initialize some variables
        posterior_cov_mat_y = None

        return cls(
            x_train,
            y_train,
            num_posterior_samples,
            num_inducing_points,
            model,
            optimizer,
            num_epochs,
            random_seed,
            num_samples_stats,
            posterior_cov_mat_y,
        )

    def train(self):
        """Train the variational by minimizing the variational loss in
        variational EM step."""
        np.random.seed(self.random_seed)
        _logger.info("# ---- Train the GPFlow model ----- #\n")
        for epoch in range(1, self.num_epochs + 1):
            self.optimizer()
            if epoch % 20 == 0 and epoch > 0:
                data = (self.x_train, self.y_train)
                loss_fun = self.model.training_loss_closure(data)
                _logger.info(
                    'Progress: %.2f %%, Epoch %s, Loss: %.2f',
                    epoch / self.num_epochs * 100,
                    epoch,
                    loss_fun().numpy(),
                )

    def predict(self, x_test, support=None, full_cov=False):
        """Predict the posterior distribution at Xnew with respect to the data
        'y'.

        Args:
            support (str): Predict wrt to latent variable f or y; not needed here as only y makes
                           sense
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
        """Compute the posterior distribution at x_test with respect to the
        data 'y'.

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
        # TODO this method call is super slow as cholesky decomp gets computed in every call
        mean, variance = self.model.predict_y(x_test)
        mean = mean.numpy()
        variance = variance.numpy()
        if full_cov is True:
            posterior_samples_y = []
            posterior_samples_mean, posterior_samples_noise = self.predict_f_samples(
                x_test, self.num_samples_stats
            )
            for mean_sample, noise_sample in zip(posterior_samples_mean, posterior_samples_noise):
                posterior_samples_y.append(np.random.normal(mean_sample, np.exp(noise_sample)))

            posterior_samples_y = np.array(posterior_samples_y)
            self.posterior_cov_mat_y = np.cov(posterior_samples_y.T)
            variance = self.posterior_cov_mat_y

        output["mean"] = mean
        output["variance"] = variance

        if self.num_posterior_samples:
            output["post_samples"] = self.predict_f_samples(x_test, self.num_posterior_samples)

        return output

    def predict_f_samples(self, x_test, num_samples):
        """Produce samples from the posterior latent function Xnew.

        Args:
            x_test (np.array):  Inputs at which to evaluate latent function f
            num_samples (int):  Number of posterior realizations of GP

        Returns:
            post_samples (np.array): Posterior samples of latent functions at x_test,
                                     (the latter might be a vector/matrix of points)
        """
        np.random.seed(self.random_seed)
        post_samples = self.model.predict_f_samples(x_test, num_samples).numpy()

        return post_samples[:, :, 0], post_samples[:, :, 1]

    @classmethod
    def _build_model(cls, num_inducing_points, x_train):
        """Build the GPflow heteroskedastic GP model.

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

        _logger.info('The GPFlow model used in this analysis is constructed as follows:')
        print_summary(model)
        _logger.info("\n")
        return model

    @classmethod
    def _build_optimizer(cls, x_train, y_train, gpflow_model, adams_training_rate, random_seed):
        """
        Build the optimization step for the variational EM step
        Args:
            x_train (np.array): Training input points
            y_train (np.array): Training output points
            gpflow_model (obj): Instance of a GPflow model
            random_seed (int): Random seed to make optimization reproducible

        Returns:
            optimization_step_fun (obj): Tensorflow optimization EM-step function for variational
                                         optimization of the GP

        """
        # TODO pull below out to json
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
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
            """Two step variational Expectation-Maximization routine for latent
            variable GPs.

            Returns:
                None
            """
            natgrad_opt.minimize(loss_fn, variational_vars)
            adam_opt.minimize(loss_fn, adam_vars)

        return optimization_step_fun
