"""A precompiled version of Gaussian Process regression."""

import logging

import numpy as np
from numba import jit
from numpy.linalg.linalg import cholesky

from pqueens.regression_approximations.regression_approximation import RegressionApproximation
from pqueens.utils.random_process_scaler import Scaler
from pqueens.utils.stochastic_optimizer import from_config_create_optimizer

_logger = logging.getLogger(__name__)

try:
    from pqueens.visualization.gnuplot_vis import gnuplot_gp_convergence
except:

    def print_import_warning(*_dummy_args):
        """Warning for gnuplotlib."""
        _logger.warning("Cannot import gnuplotlib! No terminal plots available...")

    gnuplot_gp_convergence = print_import_warning


class GPPrecompiled(RegressionApproximation):
    """A custom Gaussian process implementation using numba.

    It precompiles linear algebra operations. The GP also allows to specify a Gamma hyper-
    prior or the length scale, but only computes the MAP estimate and does not
    marginalize the hyper-parameters.

    Attributes:
        x_train_vec: TODO_doc
        y_train_vec (np.array): Training outputs for the GP.
        k_mat_inv (np.array): Inverse of the assembled covariance matrix.
        cholesky_k_mat (np.array): Lower Cholesky decomposition of the covariance matrix.
        k_mat (np.array): Assembled covariance matrix of the GP.
        gamma_k_prior (float): Parameter of Gamma hyper-prior for length scale.
        gamma_theta_prior (float): Parameter of Gamma hyper-prior for length scale.
        partial_sigma_0_sq (float): Partial derivative of evidence w.r.t. the signal variance.
        partial_l_scale_sq (float): Partial derivative of evidence w.r.t. the squared length scale.
        partial_sigma_n_sq (float): Partial derivative of evidence w.r.t. the noise variance.
        prior_mean_function_type (str): Type of mean function.
        hyper_prior_bool (bool): Boolean for lengthscale hyper-prior. If *True*, the hyperprior
                                 is used in the computations.
        stochastic_optimizer (obj): Stochastic optimizer object.
        scaler_x (obj): Scaler for inputs.
        scaler_y (obj): Scaler for outputs.
        grad_log_evidence_value (np.array): Current gradient of the log marginal likelihood w.r.t.
                                            the parameterization.
        sigma_0_sq (float): Signal variance of the RBF kernel.
        l_scale_sq (float): Squared length scale of the RBF kernel.
        sigma_n_sq (float): Noise variance of the RBF kernel.
        noise_var_lb (float): Lower bound for Gaussian noise variance in RBF kernel.
        plot_refresh_rate (int): Refresh rate of the plot (every n-iterations).
    """

    def __init__(
        self,
        x_train_vec,
        y_train_vec,
        gamma_k_prior,
        gamma_theta_prior,
        prior_mean_function_type,
        k_mat_inv,
        cholesky_k_mat,
        k_mat,
        partial_sigma_0_sq,
        partial_l_scale_sq,
        partial_sigma_n_sq,
        hyper_prior_bool,
        stochastic_optimizer,
        scaler_x,
        scaler_y,
        grad_log_evidence_value,
        sigma_0_sq,
        l_scale_sq,
        sigma_n_sq,
        noise_var_lb,
        plot_refresh_rate,
    ):
        """Instantiate the precompiled Gaussian Process.

        Args:
            x_train_vec: TODO_doc
            y_train_vec: TODO_doc
            gamma_k_prior: TODO_doc
            gamma_theta_prior: TODO_doc
            prior_mean_function_type: TODO_doc
            k_mat_inv: TODO_doc
            cholesky_k_mat: TODO_doc
            k_mat: TODO_doc
            partial_sigma_0_sq: TODO_doc
            partial_l_scale_sq: TODO_doc
            partial_sigma_n_sq: TODO_doc
            hyper_prior_bool: TODO_doc
            stochastic_optimizer: TODO_doc
            scaler_x: TODO_doc
            scaler_y: TODO_doc
            grad_log_evidence_value: TODO_doc
            sigma_0_sq: TODO_doc
            l_scale_sq: TODO_doc
            sigma_n_sq: TODO_doc
            noise_var_lb: TODO_doc
            plot_refresh_rate: TODO_doc
        """
        self.x_train_vec = x_train_vec
        self.y_train_vec = y_train_vec
        self.k_mat_inv = k_mat_inv
        self.cholesky_k_mat = cholesky_k_mat
        self.k_mat = k_mat
        self.gamma_k_prior = gamma_k_prior
        self.gamma_theta_prior = gamma_theta_prior
        self.partial_sigma_0_sq = partial_sigma_0_sq
        self.partial_l_scale_sq = partial_l_scale_sq
        self.partial_sigma_n_sq = partial_sigma_n_sq
        self.prior_mean_function_type = prior_mean_function_type
        self.hyper_prior_bool = hyper_prior_bool
        self.stochastic_optimizer = stochastic_optimizer
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.grad_log_evidence_value = grad_log_evidence_value
        self.sigma_0_sq = sigma_0_sq
        self.l_scale_sq = l_scale_sq
        self.sigma_n_sq = sigma_n_sq
        self.noise_var_lb = noise_var_lb
        self.plot_refresh_rate = plot_refresh_rate

    @classmethod
    def from_config_create(cls, config, approx_name, x_train, y_train):
        """Instantiate class GPPrecompiled from problem description.

        Args:
            config (dict): Problem description of the QUEENS analysis
            approx_name (str): Name of the regression model in input file
            x_train (np.array): Input training data for the regression model
            y_train (np.array): Output training data for the regression model

        Returns:
            Instance of GPPrecompiled class
        """
        # get the initial hyperparameters
        sigma_0_sq = config[approx_name].get("sigma_0_sq")
        if sigma_0_sq is None:
            raise ValueError(
                "The hyper-parameter 'sigma_0_sq' was not initialized in the input file. Abort..."
            )

        l_scale_sq = config[approx_name].get("l_scale_sq")
        if l_scale_sq is None:
            raise ValueError(
                "The hyper-parameter 'l_scale_sq' was not initialized in the input file. Abort..."
            )

        sigma_n_sq = config[approx_name].get("sigma_n_sq")
        if sigma_n_sq is None:
            raise ValueError(
                "The hyper-parameter 'sigma_n_sq' was not initialized in the input file. Abort..."
            )

        # get hyper prior settings
        delta_y = np.max(y_train) - np.min(y_train)
        hyper_prior_settings = config[approx_name].get("hyper_prior_lengthscale")

        hyper_prior_bool = hyper_prior_settings.get("hyper_prior_bool")
        fraction_standard_deviation = hyper_prior_settings.get("fraction_standard_deviation")
        fraction_mean_value = hyper_prior_settings.get("fraction_mean_value")

        # get normalization bool
        scaler_settings = config[approx_name].get("data_scaling")
        if scaler_settings:
            pass
        else:
            scaler_settings = {"type": "identity_scaler"}

        scaler_obj_x = Scaler.from_config_create_scaler(scaler_settings)
        scaler_obj_y = Scaler.from_config_create_scaler(scaler_settings)

        dim_y = y_train.shape[0]
        x_train = x_train.reshape(dim_y, -1)
        scaler_obj_x.fit(x_train.T)
        x_train = scaler_obj_x.transform(x_train.T).T
        scaler_obj_y.fit(y_train)
        y_train = scaler_obj_y.transform(y_train)

        # calculate gamma hyper prior parameters
        if hyper_prior_bool:
            k_prior = (
                0.25
                * (
                    fraction_mean_value / fraction_standard_deviation
                    + np.sqrt((fraction_mean_value / fraction_standard_deviation) ** 2 + 4)
                )
                ** 2
            )
            theta_prior = fraction_standard_deviation * delta_y / np.sqrt(k_prior)

        else:
            k_prior = None
            theta_prior = None

        # Check prior mean function type here as cannot be checked in precompiled jit methods
        prior_mean_function_type = config[approx_name].get("prior_mean_function_type")
        if prior_mean_function_type == "lf-identity":
            pass
        elif prior_mean_function_type is None:
            pass
        elif prior_mean_function_type == "linear":
            raise NotImplementedError(
                "A parameterized linear prior mean function is not implemented, yet. Abort..."
            )
        elif prior_mean_function_type is None:
            pass
        else:
            raise ValueError(
                f"You selected the prior mean function type '{prior_mean_function_type}', "
                f"which is not a valid option! Valid options are: 'lf-identity', 'linear' or"
                f" 'null/None'. Abort..."
            )

        # configure the stochastic optimizer and iterative averaging
        stochastic_optimizer = from_config_create_optimizer(config, approx_name)

        # get the plot refresh rate
        plot_refresh_rate = config[approx_name].get("plot_refresh_rate", None)

        # get lower bound for Gaussian noise variance in RB kernel
        noise_var_lb = config[approx_name].get("noise_var_lb")

        # Initiate some attributes
        k_mat_inv = None
        cholesky_k_mat = None
        k_mat = None
        partial_sigma_0_sq = None
        partial_l_scale_sq = None
        partial_sigma_n_sq = None
        grad_log_evidence_value = None

        return cls(
            x_train,
            y_train,
            k_prior,
            theta_prior,
            prior_mean_function_type,
            k_mat_inv,
            cholesky_k_mat,
            k_mat,
            partial_sigma_0_sq,
            partial_l_scale_sq,
            partial_sigma_n_sq,
            hyper_prior_bool,
            stochastic_optimizer,
            scaler_obj_x,
            scaler_obj_y,
            grad_log_evidence_value,
            sigma_0_sq,
            l_scale_sq,
            sigma_n_sq,
            noise_var_lb,
            plot_refresh_rate,
        )

    @staticmethod
    @jit(nopython=True)
    def pre_compile_linalg_gp(x_train_mat, sigma_0_sq, l_scale_sq, sigma_n_sq):
        r"""Pre-compile the generation of the covariance matrix.

        Also compute/pre-compile necessary derivatives for finding the MAP
        estimate of the GP. The covariance function here is the squared
        exponential covariance function.

        Args:
            x_train_mat (np.array): Training input points for the GP. Row-wise samples are stored,
                                    different columns correspond to different input dimensions
            sigma_0_sq (float): Signal standard deviation of the squared exponential covariance
                                function
            l_scale_sq (float): Length scale of the squared exponential covariance function
            sigma_n_sq (float): Gaussian noise standard deviation of the covariance function

        Returns:
            k_mat (np.array): Assembled covariance matrix of the GP
            k_mat_inv (np.array): Inverse of the assembled covariance matrix
            cholesky_k_mat (np.array): Lower Cholesky decomposition of the covariance matrix
            partial_sigma_0_sq (np.ar:): Derivative of the covariance function w.r.t.
            :math:`\sigma_0^2`
            partial_l_scale_sq (np.array): Derivative of the covariance function w.r.t.
            :math:`l_{scale}^2`
            partial_sigma_n_sq (np.array): Derivative of the covariance function w.r.t.
            :math:`\sigma_n^2`
        """
        k_mat = np.zeros((x_train_mat.shape[0], x_train_mat.shape[0]), dtype=np.float64)
        partial_l_scale_sq = np.zeros(
            (x_train_mat.shape[0], x_train_mat.shape[0]), dtype=np.float64
        )
        partial_sigma_0_sq = np.zeros(
            (x_train_mat.shape[0], x_train_mat.shape[0]), dtype=np.float64
        )

        for i, x in enumerate(x_train_mat):
            for j, y in enumerate(x_train_mat):
                if i == j:
                    noise_var = sigma_n_sq
                else:
                    noise_var = 0.0

                delta = np.linalg.norm(x - y)
                k_mat[i, j] = sigma_0_sq * np.exp(-(delta**2) / (2.0 * l_scale_sq)) + noise_var
                partial_l_scale_sq[i, j] = (
                    sigma_0_sq
                    * np.exp(-(delta**2) / (2.0 * l_scale_sq))
                    * delta**2
                    / (2.0 * l_scale_sq**2)
                )
                partial_sigma_0_sq[i, j] = np.exp(-(delta**2) / (2.0 * l_scale_sq))

        k_mat_inv = np.linalg.inv(k_mat)
        cholesky_k_mat = cholesky(k_mat)

        partial_sigma_n = np.eye(k_mat.shape[0])

        return (
            k_mat,
            k_mat_inv,
            cholesky_k_mat,
            partial_sigma_0_sq,
            partial_l_scale_sq,
            partial_sigma_n,
        )

    @staticmethod
    @jit(nopython=True)
    def posterior_mean(
        k_mat_inv,
        x_test_mat,
        x_train_mat,
        y_train_vec,
        sigma_0_sq,
        l_scale_sq,
        prior_mean_function_type,
    ):
        """Precompile the posterior mean function of the Gaussian Process.

        Args:
            k_mat_inv (np.array): Inverse of the assembled covariance matrix
            x_test_mat (np.array): Testing input points for the GP. Individual samples row-wise,
                        columns correspond to different dimensions
            x_train_mat (np.array): Training input points for the GP. Individual samples row-wise,
                                    columns correspond to different dimensions
            y_train_vec (np.array): Training outputs for the GP. Column vector where rows correspond
                                    to different samples
            sigma_0_sq (float): Signal variance for squared exponential covariance function
            l_scale_sq (float): Squared length scale for squared exponential covariance function
            prior_mean_function_type (str): Type of the prior mean function

        Returns:
            mu_vec (np.array): Posterior mean vector of the Gaussian Process evaluated at
            *x_test_vec*
        """
        k_vec = np.zeros((x_train_mat.shape[0], x_test_mat.shape[0]), dtype=np.float64)
        for j, x_test in enumerate(x_test_mat):
            for i, x_train in enumerate(x_train_mat):
                delta = np.linalg.norm(x_test - x_train)
                k_vec[i, j] = sigma_0_sq * np.exp(-(delta**2) / (2 * l_scale_sq))

        if prior_mean_function_type is None:
            mu_vec = np.dot(np.dot(k_vec.T, k_mat_inv), (y_train_vec))
        elif prior_mean_function_type == "lf-identity":

            mean_train_prior = x_train_mat
            mean_test_prior = x_test_mat

            # please note that the first column holds the output values of the LF
            # model in this formulation.
            mu_vec = mean_test_prior[:, 0] + np.dot(
                np.dot(k_vec.T, k_mat_inv), (y_train_vec - mean_train_prior[:, 0])
            )

        return mu_vec

    @staticmethod
    @jit(nopython=True)
    def posterior_var(
        k_mat_inv,
        x_test_mat,
        x_train_mat,
        sigma_0_sq,
        l_scale_sq,
        sigma_n_sq,
        support,
    ):
        """Precompile the posterior variance function of the Gaussian Process.

        Args:
            k_mat_inv (np.array): Inverse of the assembled covariance matrix
            x_test_mat (np.array): Testing input points for the GP. Individual samples row-wise,
                        columns correspond to different dimensions
            x_train_mat (np.array): Training input points for the GP. Individual samples row-wise,
                                    columns correspond to different dimensions
            sigma_0_sq (float): Signal variance for squared exponential covariance function
            l_scale_sq (float): Squared length scale for squared exponential covariance function
            sigma_n_sq: TODO_doc
            support (str): Support type for the posterior distribution. For 'y' the posterior
                        is computed w.r.t. the output data; for 'f' the GP is computed w.r.t. the
                        latent function `f`

        Returns:
            posterior_variance_vec (np.array): Posterior variance vector of the GP evaluated
            at the testing points *x_test_vec*
        """
        k_mat_test_train = np.zeros((x_train_mat.shape[0], x_test_mat.shape[0]), dtype=np.float64)
        for j, x_test in enumerate(x_test_mat):
            for i, x_train in enumerate(x_train_mat):
                delta = np.linalg.norm(x_test - x_train)
                k_mat_test_train[i, j] = sigma_0_sq * np.exp(-(delta**2) / (2 * l_scale_sq))

        k_mat_test = np.zeros((x_test_mat.shape[0], x_test_mat.shape[0]), dtype=np.float64)
        for j, x_test1 in enumerate(x_test_mat):
            for i, x_test2 in enumerate(x_test_mat):
                delta = np.linalg.norm(x_test1 - x_test2)
                k_mat_test[i, j] = sigma_0_sq * np.exp(-(delta**2) / (2 * l_scale_sq))

        posterior_variance_vec = np.diag(
            k_mat_test - np.dot(np.dot(k_mat_test_train.T, k_mat_inv), k_mat_test_train)
        )

        if support == 'y':
            posterior_variance_vec = posterior_variance_vec + sigma_n_sq

        return posterior_variance_vec

    @staticmethod
    @jit(nopython=True)
    def grad_log_evidence(
        param_vec,
        y_train_vec,
        x_train_vec,
        k_mat_inv,
        partial_sigma_0_sq,
        partial_l_scale_sq,
        partial_sigma_n_sq,
    ):
        """Calculate gradient of log-evidence.

        Gradient of the log evidence function of the GP w.r.t. the
        variational hyperparameters. The latter might be a transformed
        representation of the actual hyperparameters.

        Args:
            param_vec (np.array): Vector containing values of hyper-parameters.
                                    **Note**: This is already used here in some of the other input
                                    values are computed beforehand and stored as attributes
                                    (**TODO_doc:** reformulate the note.)
            y_train_vec (np.array): Output training vector of the GP
            x_train_vec (np.array): Input training vector for the GP
            k_mat_inv (np.array): Current inverse of the GP covariance matrix
            partial_sigma_0_sq (np.array): Partial derivative of covariance matrix w.r.t. signal
                                            variance variational parameter
            partial_l_scale_sq (np.array): Partial derivative of covariance matrix w.r.t. length
                                        squared scale variational parameter
            partial_sigma_n_sq (np.array): Partial derivative of covariance matrix w.r.t. noise
                                            variance variational parameter

        Returns:
            grad (np.array): Gradient vector of the evidence w.r.t. the parameterization
            of the hyperparameters
        """
        sigma_0_sq_param, l_scale_sq_param, sigma_n_sq_param = param_vec
        data_minus_prior_mean = y_train_vec - x_train_vec
        alpha = np.dot(k_mat_inv, data_minus_prior_mean)

        grad_ev_sigma_0_sq_param = 0.5 * np.trace(
            (np.dot(alpha, alpha.T) - k_mat_inv) @ (partial_sigma_0_sq) * np.exp(sigma_0_sq_param)
        )
        grad_ev_sigma_n_sq_param = 0.5 * np.trace(
            (np.dot(alpha, alpha.T) - k_mat_inv) @ (partial_sigma_n_sq) * np.exp(sigma_n_sq_param)
        )
        grad_ev_l_scale_sq_param = 0.5 * np.trace(
            (np.dot(alpha, alpha.T) - k_mat_inv) @ (partial_l_scale_sq) * np.exp(l_scale_sq_param)
        )
        grad = np.array(
            [grad_ev_sigma_0_sq_param, grad_ev_l_scale_sq_param, grad_ev_sigma_n_sq_param]
        ).flatten()

        return grad

    def log_evidence(self):
        """Log evidence/log marginal likelihood of the GP.

        Returns:
            evidence_eff (float): Evidence of the GP for current choice of
            hyperparameters
        """
        # decide which mean prior formulation shall be used
        y_dim = self.y_train_vec.flatten().size
        if self.prior_mean_function_type is None:
            data_minus_prior_mean = self.y_train_vec.reshape(y_dim, 1)
        elif self.prior_mean_function_type == 'lf-identity':
            data_minus_prior_mean = (
                self.y_train_vec.reshape(y_dim, 1) - self.x_train_vec[:, 0, None]
            )
        elif self.prior_mean_function_type == 'linear':
            raise NotImplementedError(
                "The prior mean function type 'linear' is not implemented, yet. Abort..."
            )
        else:
            raise ValueError(
                f"Your specified {self.prior_mean_function_type} for a prior mean function type, "
                f"which is not a valid option. Valid options are: 'null/None', 'lf-identity' or "
                f"'linear'. Abort..."
            )

        # decide whether a hyper prior should be used
        if self.hyper_prior_bool:
            evidence_eff = (
                -0.5
                * np.dot(np.dot(data_minus_prior_mean.T, self.k_mat_inv), data_minus_prior_mean)
                - (np.sum(np.log(np.diag(self.cholesky_k_mat))))
                + (self.k_prior - 1) * np.log(self.l_scale_sq)
                + 1 / self.theta_prior * self.l_scale_sq
            )
        else:
            evidence_eff = (
                -0.5
                * np.dot(np.dot(data_minus_prior_mean.T, self.k_mat_inv), data_minus_prior_mean)
                - (np.sum(np.log(np.diag(self.cholesky_k_mat))))
                - self.k_mat.shape[0] / 2 * np.log(2 * np.pi)
            )
        return evidence_eff.flatten()

    def train(self):
        """Train the Gaussian Process.

        Training is conducted by maximizing the evidence/marginal
        likelihood by minimizing the negative log evidence.
        """
        # initialize hyperparameters and associated linear algebra
        sigma_0_sq_param_init = np.log(self.sigma_0_sq)
        l_sq_param_init = np.log(self.l_scale_sq)
        sigma_n_sq_param_init = np.log(self.sigma_n_sq)
        x_0 = np.array([sigma_0_sq_param_init, l_sq_param_init, sigma_n_sq_param_init])

        (
            self.k_mat,
            self.k_mat_inv,
            self.cholesky_k_mat,
            self.partial_sigma_0_sq,
            self.partial_l_scale_sq,
            self.partial_sigma_n_sq,
        ) = GPPrecompiled.pre_compile_linalg_gp(
            self.x_train_vec, self.sigma_0_sq, self.l_scale_sq, self.sigma_n_sq
        )

        _logger.info("Initiating training of the GP model...")

        # set-up stochastic optimizer
        self.stochastic_optimizer.current_variational_parameters = x_0
        self.stochastic_optimizer.gradient = lambda param_vec: GPPrecompiled.grad_log_evidence(
            param_vec,
            self.y_train_vec,
            self.x_train_vec,
            self.k_mat_inv,
            self.partial_sigma_0_sq,
            self.partial_l_scale_sq,
            self.partial_sigma_n_sq,
        )
        log_ev_max = -np.inf
        log_ev_lst = []
        iter_lst = []
        for params in self.stochastic_optimizer:
            rel_L2_change_params = self.stochastic_optimizer.rel_L2_change
            iteration = self.stochastic_optimizer.iteration

            # update parameters and associated linear algebra
            sigma_0_sq_param, l_scale_sq_param, sigma_n_sq_param = params
            self.sigma_0_sq = np.exp(sigma_0_sq_param)
            self.l_scale_sq = np.exp(l_scale_sq_param)
            self.sigma_n_sq = max(np.exp(sigma_n_sq_param), self.noise_var_lb)

            self.grad_log_evidence_value = self.stochastic_optimizer.current_gradient_value
            (
                self.k_mat,
                self.k_mat_inv,
                self.cholesky_k_mat,
                self.partial_sigma_0_sq,
                self.partial_l_scale_sq,
                self.partial_sigma_n_sq,
            ) = GPPrecompiled.pre_compile_linalg_gp(
                self.x_train_vec,
                self.sigma_0_sq,
                self.l_scale_sq,
                self.sigma_n_sq,
            )
            self.stochastic_optimizer.gradient = lambda param_vec: GPPrecompiled.grad_log_evidence(
                param_vec,
                self.y_train_vec,
                self.x_train_vec,
                self.k_mat_inv,
                self.partial_sigma_0_sq,
                self.partial_l_scale_sq,
                self.partial_sigma_n_sq,
            )

            log_ev = self.log_evidence()

            iter_lst.append(iteration)
            log_ev_lst.append(log_ev)

            if self.plot_refresh_rate:
                if iteration % int(self.plot_refresh_rate) == 0:

                    # make some funky gnuplot terminal plots
                    gnuplot_gp_convergence(iter_lst, log_ev_lst)

                    # Verbose output
                    _logger.info(
                        "Iter %s, parameters %s, gradient log evidence: "
                        "%s, rel L2 change "
                        "%.6f, log-evidence: %s",
                        iteration,
                        params,
                        self.grad_log_evidence_value,
                        rel_L2_change_params,
                        log_ev,
                    )

            # store the max value for log evidence along with the parameters
            if log_ev > log_ev_max:
                log_ev_max = log_ev
                params_ev_max = params
                k_mat_ev_max = self.k_mat
                k_mat_inv_ev_max = self.k_mat_inv
                cholesky_k_mat_ev_max = self.cholesky_k_mat

        # use the params that yielded the max log evidence
        sigma_0_sq_param, l_scale_sq_param, sigma_n_sq_param = params_ev_max
        self.sigma_0_sq = np.exp(sigma_0_sq_param)
        self.l_scale_sq = np.exp(l_scale_sq_param)
        self.sigma_n_sq = max(np.exp(sigma_n_sq_param), self.noise_var_lb)
        self.k_mat = k_mat_ev_max
        self.k_mat_inv = k_mat_inv_ev_max
        self.cholesky_k_mat = cholesky_k_mat_ev_max

        _logger.info("GP model trained sucessfully!")

    def predict(self, x_test_mat, support='f'):
        """Predict the posterior distribution of the trained GP at *x_test*.

        Args:
            x_test_mat (np.array): Testing matrix for GP with row-wise (vector-valued)
                                   testing points
            support (str): Type of support for which the GP posterior is computed; If:

                            * 'f': Posterior w.r.t. the latent function `f`
                            * 'y': Latent function is marginalized such that posterior is defined
                              w.r.t. the output 'y' (introduces extra variance)

        Returns:
            output (dict): Output dictionary containing the posterior of the GP
        """
        dim_y = self.y_train_vec.size
        dim_x = self.x_train_vec.reshape(dim_y, -1).shape[1]

        posterior_mean_test_vec = GPPrecompiled.posterior_mean(
            self.k_mat_inv,
            self.scaler_x.transform(x_test_mat.reshape(-1, dim_x)),
            self.x_train_vec,
            self.y_train_vec.flatten(),
            self.sigma_0_sq,
            self.l_scale_sq,
            self.prior_mean_function_type,
        )

        var = GPPrecompiled.posterior_var(
            self.k_mat_inv,
            self.scaler_x.transform(x_test_mat.reshape(-1, dim_x)),
            self.x_train_vec.reshape(dim_y, -1),
            self.sigma_0_sq,
            self.l_scale_sq,
            self.sigma_n_sq,
            support,
        )
        assert np.any(var.flatten() > 0.0), (
            'Posterior variance has negative values! It seems like the condition of your '
            'covariance matrix is rather bad. Please increase the noise variance lower bound! '
            'Abort....'
        )

        output = {"x_test": x_test_mat.reshape(-1, dim_x)}
        output["mean"] = self.scaler_y.inverse_transform_mean(posterior_mean_test_vec)
        output["variance"] = self.scaler_y.inverse_transform_std(np.sqrt(var)) ** 2

        return output

    def get_state(self):
        """Get the current hyper-parameters of the model.

        Returns:
            state_dict (dict): Dictionary with the current state settings
            of the probabilistic mapping object
        """
        hyper_params_dict = {
            'sigma_0_sq': self.sigma_0_sq,
            'l_scale_sq': self.l_scale_sq,
            'sigma_n_sq': self.sigma_n_sq,
            'k_mat': self.k_mat,
            'k_mat_inv': self.k_mat_inv,
            'cholesky_k_mat': self.cholesky_k_mat,
        }
        return hyper_params_dict

    def set_state(self, state_dict):
        """Update and set new hyper-parameters for the model.

        Args:
            state_dict (dict): Dictionary with the current state settings
                               of the probabilistic mapping object
        """
        # conduct some checks
        valid_keys = [
            'sigma_0_sq',
            'l_scale_sq',
            'sigma_n_sq',
            'k_mat',
            'k_mat_inv',
            'cholesky_k_mat',
        ]
        assert isinstance(
            state_dict, dict
        ), "The provided state_dict must be a dictionary! Abort..."
        keys = list(state_dict.keys())
        assert keys == valid_keys, "The provided dictionary does not contain valid keys! Abort..."

        # Actually set the new state of the object
        self.sigma_0_sq = state_dict['sigma_0_sq']
        self.l_scale_sq = state_dict['l_scale_sq']
        self.sigma_n_sq = state_dict['sigma_n_sq']
        self.k_mat = state_dict['k_mat']
        self.k_mat_inv = state_dict['k_mat_inv']
        self.cholesky_k_mat = state_dict['cholesky_k_mat']
