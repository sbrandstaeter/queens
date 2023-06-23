"""Variational distribution utils."""

# pylint: disable=too-many-lines
# pylint: disable=invalid-name
import abc

import numpy as np
import scipy
from numba import njit

from pqueens.distributions.particles import ParticleDiscreteDistribution


class VariationalDistribution:
    """Base class for probability distributions for variational inference.

    Attributes:
        dimension (int): dimension of the distribution
    """

    def __init__(self, dimension):
        """Initialize variational distribution."""
        self.dimension = dimension

    @abc.abstractmethod
    def construct_variational_parameters(self):
        """Construct the variational parameters from distribution parameters.

        The inputs to this methods are the parameters that would be needed to construct a QUEENS
        distribution object, e.g. for a Gaussian distribution the inputs are mean and covariance

        Returns:
            variational_parameters (np.ndarray): Variational parameters
        """

    @abc.abstractmethod
    def reconstruct_distribution_parameters(self, variational_parameters):
        """Reconstruct distribution parameters from variational parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters
        """

    @abc.abstractmethod
    def draw(self, variational_parameters, num_draws=1):
        """Draw *num_draws* samples from distribution.

        Args:
           variational_parameters (np.ndarray):  variational parameters (1 x n_params)
           num_draws (int): Number of samples
        """

    @abc.abstractmethod
    def logpdf(self, variational_parameters, x):
        """Evaluate the natural logarithm of the logpdf at sample.

        Args:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
            x (np.ndarray): Locations to evaluate (n_samples x n_dim)
        """

    @abc.abstractmethod
    def pdf(self, variational_parameters, x):
        """Evaluate the probability density function (pdf) at sample.

        Args:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
            x (np.ndarray): Locations to evaluate (n_samples x n_dim)
        """

    @abc.abstractmethod
    def grad_params_logpdf(self, variational_parameters, x):
        """Logpdf gradient w.r.t. the variational parameters.

        Evaluated at samples  *x*. Also known as the score function.

        Args:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
            x (np.ndarray): Locations to evaluate (n_samples x n_dim)
        """

    @abc.abstractmethod
    def fisher_information_matrix(self, variational_parameters):
        """Compute the fisher information matrix.

        Depends on the variational distribution for the given
        parameterization.

        Args:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
        """

    @abc.abstractmethod
    def initialize_variational_parameters(self, random=False):
        """Initialize variational parameters.

        Args:
            random (bool, optional): If True, a random initialization is used. Otherwise the
                                     default is selected

        Returns:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
        """

    @abc.abstractmethod
    def export_dict(self, variational_parameters):
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            export_dict (dictionary): Dict containing distribution information
        """


class MeanFieldNormalVariational(VariationalDistribution):
    r"""Mean field multivariate normal distribution.

    Uses the parameterization (as in [1]):  :math:`parameters=[\mu, \lambda]`
    where :math:`\mu` are the mean values and :math:`\sigma^2=exp(2 \lambda)`
    the variances allowing for :math:`\lambda` to be unconstrained.

    References:
        [1]: Kucukelbir, Alp, et al. "Automatic differentiation variational inference."
             The Journal of Machine Learning Research 18.1 (2017): 430-474.

    Attributes:
        n_parameters (int): Number of parameters used in the parameterization.
    """

    def __init__(self, dimension):
        """Initialize variational distribution.

        Args:
            dimension (int): Dimension of RV.
        """
        super().__init__(dimension)
        self.n_parameters = 2 * dimension

    @abc.abstractmethod
    def initialize_variational_parameters(self, random=False):
        r"""Initialize variational parameters.

        Default initialization:
            :math:`\mu=0` and :math:`\sigma^2=1`

        Random intialization:
            :math:`\mu=Uniform(-0.1,0.1)` and :math:`\sigma^2=Uniform(0.9,1.1)`

        Args:
            random (bool, optional): If True, a random initialization is used. Otherwise the
                                     default is selected

        Returns:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
        """
        if random:
            variational_parameters = np.hstack(
                (
                    0.1 * (-0.5 + np.random.rand(self.dimension)),
                    0.5 + np.log(1 + 0.1 * (-0.5 + np.random.rand(self.dimension))),
                )
            )
        else:
            variational_parameters = np.zeros(self.n_parameters)

        return variational_parameters

    @staticmethod
    def construct_variational_parameters(mean, covariance):  # pylint: disable=arguments-differ
        """Construct the variational parameters from mean and covariance.

        Args:
            mean (np.ndarray): Mean values of the distribution (n_dim x 1)
            covariance (np.ndarray): Covariance matrix of the distribution (n_dim x n_dim)

        Returns:
            variational_parameters (np.ndarray): Variational parameters
        """
        if len(mean) == len(covariance):
            variational_parameters = np.hstack((mean.flatten(), 0.5 * np.log(np.diag(covariance))))
        else:
            raise ValueError(
                f"Dimension of the mean value {len(mean)} does not equal covariance dimension"
                f"{covariance.shape}"
            )
        return variational_parameters

    def reconstruct_distribution_parameters(self, variational_parameters):
        """Reconstruct mean and covariance from the variational parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            mean (np.ndarray): Mean value of the distribution (n_dim x 1)
            cov (np.ndarray): Covariance matrix of the distribution (n_dim x n_dim)
        """
        mean, cov = (
            variational_parameters[: self.dimension],
            np.exp(2 * variational_parameters[self.dimension :]),
        )
        return mean.reshape(-1, 1), np.diag(cov)

    def _grad_reconstruct_distribution_parameters(self, variational_parameters):
        """Gradient of the parameter reconstruction.

         Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            grad_reconstruct_params (np.ndarray): Gradient vector of the reconstruction
                                                w.r.t. the variational parameters
        """
        grad_mean = np.ones((1, self.dimension))
        grad_std = (np.exp(variational_parameters[self.dimension :])).reshape(1, -1)
        grad_reconstruct_params = np.hstack((grad_mean, grad_std))
        return grad_reconstruct_params

    def draw(self, variational_parameters, num_draws=1):
        """Draw *num_draw* samples from the variational distribution.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            num_draw (int): Number of samples to draw

        Returns:
            samples (np.ndarray): Row-wise samples of the variational distribution
        """
        mean, cov = self.reconstruct_distribution_parameters(variational_parameters)
        samples = np.random.randn(num_draws, self.dimension) * np.sqrt(np.diag(cov)).reshape(
            1, -1
        ) + mean.reshape(1, -1)
        return samples

    def logpdf(self, variational_parameters, x):
        """Logpdf evaluated using the variational parameters at samples `x`.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            logpdf (np.ndarray): Row vector of the logpdfs
        """
        mean, cov = self.reconstruct_distribution_parameters(variational_parameters)
        mean = mean.flatten()
        cov = np.diag(cov)
        x = np.atleast_2d(x)
        logpdf = (
            -0.5 * self.dimension * np.log(2 * np.pi)
            - np.sum(variational_parameters[self.dimension :])
            - 0.5 * np.sum((x - mean) ** 2 / cov, axis=1)
        )
        return logpdf.flatten()

    def pdf(self, variational_parameters, x):
        """Pdf of the variational distribution evaluated at samples *x*.

        First computes the logpdf, which is numerically more stable for exponential distributions.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            pdf (np.ndarray): Row vector of the pdfs
        """
        pdf = np.exp(self.logpdf(variational_parameters, x))
        return pdf

    def grad_params_logpdf(self, variational_parameters, x):
        """Logpdf gradient w.r.t. the variational parameters.

        Evaluated at samples *x*. Also known as the score function.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            score (np.ndarray): Column-wise scores
        """
        mean, cov = self.reconstruct_distribution_parameters(variational_parameters)
        mean = mean.flatten()
        cov = np.diag(cov)
        dlnN_dmu = (x - mean) / cov
        dlnN_dsigma = (x - mean) ** 2 / cov - np.ones(x.shape)
        score = np.concatenate(
            [
                dlnN_dmu.T.reshape(self.dimension, len(x)),
                dlnN_dsigma.T.reshape(self.dimension, len(x)),
            ]
        )
        return score

    def grad_logpdf_sample(self, sample_batch, variational_parameters):
        """Computes the gradient of the logpdf w.r.t. *x*.

        Args:
            sample_batch (np.ndarray): Row-wise samples
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            gradients_batch (np.ndarray): Gradients of the log-pdf w.r.t. the
            sample *x*. The first dimension of the array corresponds to
            the different samples. The second dimension to different dimensions
            within one sample. (Third dimension is empty and just added to
            keep slices two dimensional.)
        """
        mean, cov = self.reconstruct_distribution_parameters(variational_parameters)
        mean = mean.flatten()
        cov = np.diag(cov)
        gradient_lst = []
        for sample in sample_batch:
            gradient_lst.append((-(sample - mean) / cov).reshape(-1, 1))

        gradients_batch = np.array(gradient_lst)
        return gradients_batch

    def fisher_information_matrix(self, variational_parameters):
        r"""Compute the Fisher information matrix analytically.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            FIM (np.ndarray): Matrix (num parameters :math:`\times` num parameters)
        """
        fisher_diag = np.exp(-2 * variational_parameters[self.dimension :])
        fisher_diag = np.hstack((fisher_diag, 2 * np.ones(self.dimension)))
        return np.diag(fisher_diag)

    def export_dict(self, variational_parameters):
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            export_dict (dictionary): Dict containing distribution information
        """
        mean, cov = self.reconstruct_distribution_parameters(variational_parameters)
        sd = cov**0.5
        export_dict = {
            "type": "meanfield_Normal",
            "mean": mean,
            "covariance": cov,
            "standard_deviation": sd,
            "variational_parameters": variational_parameters,
        }
        return export_dict

    def conduct_reparameterization(self, variational_parameters, n_samples):
        """Conduct a reparameterization.

        Args:
            variational_parameters (np.ndarray): Array with variational parameters
            n_samples (int): Number of samples for current batch

        Returns:
            * samples_mat (np.ndarray): Array of actual samples from the
              variational distribution
            * standard_normal_sample_batch (np.ndarray): Standard normal
              distributed sample batch
        """
        standard_normal_sample_batch = np.random.normal(0, 1, size=(n_samples, self.dimension))
        mean, cov = self.reconstruct_distribution_parameters(variational_parameters)
        samples_mat = mean.flatten() + np.sqrt(np.diag(cov)) * standard_normal_sample_batch

        return samples_mat, standard_normal_sample_batch

    def jacobi_variational_parameters_reparameterization(
        self, standard_normal_sample_batch, variational_parameters
    ):
        r"""Calculate the gradient of the reparameterization.

        Args:
            standard_normal_sample_batch (np.ndarray): Standard normal distributed sample
                                                    batch
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            jacobi_reparameterization_batch (np.ndarray): Tensor with Jacobi matrices
            for the reparameterization trick. The first dimension loops over the
            individual samples, the second dimension over variational parameters and
            the last dimension over the dimensions within one sample.

        Note:
            We assume that *grad_reconstruct_params* is a row-vector containing the partial
            derivatives of the reconstruction mapping of the actual distribution parameters
            w.r.t. the variational parameters.

            The variable *jacobi_parameters* is the (n_parameters :math:`\times` dim_sample)
            Jacobi matrix of the reparameterization w.r.t. the distribution parameters,
            with differentiating after the distribution
            parameters in different rows and different output dimensions of the sample per
            column.
        """
        jacobi_reparameterization_lst = []
        grad_reconstruct_params = self._grad_reconstruct_distribution_parameters(
            variational_parameters
        )
        for sample in standard_normal_sample_batch:
            jacobi_parameters = np.vstack((np.eye(self.dimension), np.diag(sample)))
            jacobi_reparameterization = jacobi_parameters * grad_reconstruct_params.T
            jacobi_reparameterization_lst.append(jacobi_reparameterization)

        jacobi_reparameterization_batch = np.array(jacobi_reparameterization_lst)
        return jacobi_reparameterization_batch


class FullRankNormalVariational(VariationalDistribution):
    r"""Fullrank multivariate normal distribution.

    Uses the parameterization (as in [1])
    :math:`parameters=[\mu, \lambda]`, where :math:`\mu` are the mean values and
    :math:`\lambda` is an array containing the nonzero entries of the lower Cholesky
    decomposition of the covariance matrix :math:`L`:
    :math:`\lambda=[L_{00},L_{10},L_{11},L_{20},L_{21},L_{22}, ...]`.
    This allows the parameters :math:`\lambda` to be unconstrained.

    References:
        [1]: Kucukelbir, Alp, et al. "Automatic differentiation variational inference."
             The Journal of Machine Learning Research 18.1 (2017): 430-474.

    Attributes:
        n_parameters (int): Number of parameters used in the parameterization.
    """

    def __init__(self, dimension):
        """Initialize variational distribution.

        Args:
            dimension (int): dimension of the RV
        """
        super().__init__(dimension)
        self.n_parameters = (dimension * (dimension + 1)) // 2 + dimension

    @abc.abstractmethod
    def initialize_variational_parameters(self, random=False):
        r"""Initialize variational parameters.

        Default initialization:
            :math:`\mu=0` and :math:`L=diag(1)` where :math:`\Sigma=LL^T`

        Random intialization:
            :math:`\mu=Uniform(-0.1,0.1)` :math:`L=diag(Uniform(0.9,1.1))` where :math:`\Sigma=LL^T`

        Args:
            random (bool, optional): If True, a random initialization is used. Otherwise the
                                     default is selected

        Returns:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
        """
        if random:
            cholesky_covariance = np.eye(self.dimension) + 0.1 * (
                -0.5 + np.diag(np.random.rand(self.dimension))
            )
            variational_parameters = np.zeros(self.dimension) + 0.1 * (
                -0.5 + np.random.rand(self.dimension)
            )
            for j in range(len(cholesky_covariance)):
                variational_parameters = np.hstack(
                    (variational_parameters, cholesky_covariance[j, : j + 1])
                )
        else:
            mean = np.zeros(self.dimension)
            L = np.ones((self.dimension * (self.dimension + 1)) // 2)
            variational_parameters = np.concatenate([mean, L])

        return variational_parameters

    @staticmethod
    def construct_variational_parameters(mean, covariance):  # pylint: disable=arguments-differ
        """Construct the variational parameters from mean and covariance.

        Args:
            mean (np.ndarray): Mean values of the distribution (n_dim x 1)
            covariance (np.ndarray): Covariance matrix of the distribution (n_dim x n_dim)

        Returns:
            variational_parameters (np.ndarray): Variational parameters
        """
        if len(mean) == len(covariance):
            cholesky_covariance = np.linalg.cholesky(covariance)
            variational_parameters = mean.flatten()
            for j in range(len(cholesky_covariance)):
                variational_parameters = np.hstack(
                    (variational_parameters, cholesky_covariance[j, : j + 1])
                )
        else:
            raise ValueError(
                f"Dimension of the mean value {len(mean)} does not equal covariance dimension"
                f"{covariance.shape}"
            )
        return variational_parameters

    def reconstruct_distribution_parameters(self, variational_parameters, return_cholesky=False):
        """Reconstruct mean value, covariance and its Cholesky decomposition.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            return_cholesky (bool, optional): Return the L if desired
        Returns:
            mean (np.ndarray): Mean value of the distribution (n_dim x 1)
            cov (np.ndarray): Covariance of the distribution (n_dim x n_dim)
            L (np.ndarray): Cholesky decomposition of the covariance matrix (n_dim x n_dim)
        """
        mean = variational_parameters[: self.dimension].reshape(-1, 1)
        cholesky_covariance_array = variational_parameters[self.dimension :]
        cholesky_covariance = np.zeros((self.dimension, self.dimension))
        idx = np.tril_indices(self.dimension, k=0, m=self.dimension)
        cholesky_covariance[idx] = cholesky_covariance_array
        cov = np.matmul(cholesky_covariance, cholesky_covariance.T)

        if return_cholesky:
            return mean, cov, cholesky_covariance

        return mean, cov

    def _grad_reconstruct_distribution_parameters(self):
        """Gradient of the parameter reconstruction.

        Returns:
            grad_reconstruct_params (np.ndarray): Gradient vector of the reconstruction
                                                w.r.t. the variational parameters
        """
        grad_mean = np.ones((1, self.dimension))
        grad_cholesky = np.ones((1, self.n_parameters - self.dimension))
        grad_reconstruct_params = np.hstack((grad_mean, grad_cholesky))
        return grad_reconstruct_params

    def draw(self, variational_parameters, num_draws=1):
        """Draw *num_draw* samples from the variational distribution.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            num_draw (int): Number of samples to draw

        Returns:
            samples (np.ndarray): Row-wise samples of the variational distribution
        """
        mean, _, L = self.reconstruct_distribution_parameters(
            variational_parameters, return_cholesky=True
        )
        sample = np.dot(L, np.random.randn(self.dimension, num_draws)).T + mean.reshape(1, -1)
        return sample

    def logpdf(self, variational_parameters, x):
        """Logpdf evaluated using the at samples *x*.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            logpdf (np.ndarray): Row vector of the logpdfs
        """
        mean, cov, L = self.reconstruct_distribution_parameters(
            variational_parameters, return_cholesky=True
        )
        x = np.atleast_2d(x)
        u = np.linalg.solve(cov, (x.T - mean))

        def col_dot_prod(x, y):
            return np.sum(x * y, axis=0)

        logpdf = (
            -0.5 * self.dimension * np.log(2 * np.pi)
            - np.sum(np.log(np.abs(np.diag(L))))
            - 0.5 * col_dot_prod(x.T - mean, u)
        )
        return logpdf.flatten()

    def pdf(self, variational_parameters, x):
        """Pdf of evaluated at given samples *x*.

        First computes the logpdf, which is numerically more stable for exponential distributions.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            pdf (np.ndarray): Row vector of the pdfs
        """
        pdf = np.exp(self.logpdf(variational_parameters, x))
        return pdf

    def grad_params_logpdf(self, variational_parameters, x):
        """Logpdf gradient w.r.t. to the variational parameters.

        Evaluated at samples *x*. Also known as the score function.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            score (np.ndarray): Column-wise scores
        """
        mean, cov, L = self.reconstruct_distribution_parameters(
            variational_parameters, return_cholesky=True
        )
        x = np.atleast_2d(x)
        # Helper variable
        q = np.linalg.solve(cov, x.T - mean)
        dlnN_dmu = q.copy()
        diag_indx = np.cumsum(np.arange(1, self.dimension + 1)) - 1
        n_params_chol = (self.dimension * (self.dimension + 1)) // 2
        dlnN_dsigma = np.zeros((n_params_chol, 1))
        # Term due to determinant
        dlnN_dsigma[diag_indx] = -1.0 / (np.diag(L).reshape(-1, 1))
        dlnN_dsigma = np.tile(dlnN_dsigma, (1, len(x)))
        # Term due to normalization
        b = np.matmul(L.T, q)
        indx = 0
        f = np.zeros(dlnN_dsigma.shape)
        for r in range(0, self.dimension):
            for s in range(0, r + 1):
                dlnN_dsigma[indx, :] += q[r, :] * b[s, :]
                f[indx, :] += q[r, :] * b[s, :]
                indx += 1
        score = np.vstack((dlnN_dmu, dlnN_dsigma))
        return score

    def grad_logpdf_sample(self, sample_batch, variational_parameters):
        """Computes the gradient of the logpdf w.r.t. to the *x*.

        Args:
            sample_batch (np.ndarray): Row-wise samples
            variational_parameters (np.ndarray): Variational parameters


        Returns:
            gradients_batch (np.ndarray): Gradients of the log-pdf w.r.t. the
            sample *x*. The first dimension of the
            array corresponds to the different samples.
            The second dimension to different dimensions
            within one sample. (Third dimension is empty
            and just added to keep slices two-dimensional.)
        """
        mean, cov, _ = self.reconstruct_distribution_parameters(variational_parameters)
        gradient_lst = []
        for sample in sample_batch:
            gradient_lst.append(
                np.dot(np.linalg.inv(cov), -(sample.reshape(-1, 1) - mean)).reshape(-1, 1)
            )

        gradients_batch = np.array(gradient_lst)
        return gradients_batch

    def fisher_information_matrix(self, variational_parameters):
        """Compute the Fisher information matrix analytically.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            FIM (np.ndarray): Matrix (num parameters x num parameters)
        """
        _, cov, L = self.reconstruct_distribution_parameters(
            variational_parameters, return_cholesky=True
        )

        def fim_blocks(dimension):
            """Compute the blocks of the FIM."""
            mu_block = np.linalg.inv(cov + 1e-8 * np.eye(len(cov)))
            n_params_chol = (dimension * (dimension + 1)) // 2
            sigma_block = np.zeros((n_params_chol, n_params_chol))
            A_list = []
            # Improvements of this implementation are welcomed!
            for r in range(0, dimension):
                for s in range(0, r + 1):
                    A_q = np.zeros(L.shape)
                    A_q[r, s] = 1
                    A_q = L @ A_q.T
                    A_q = np.linalg.solve(cov, A_q + A_q.T)
                    A_list.append(A_q)
            for p in range(n_params_chol):
                for q in range(p + 1):
                    val = 0.5 * np.trace(A_list[p] @ A_list[q])
                    sigma_block[p, q] = val
                    sigma_block[q, p] = val
            return mu_block, sigma_block

        # Using jit is useful in higher dimensional cases but introduces an computational overhead
        # for lowerdimensional cases. Doing some tests showed that the break evenpoint is reached
        # at around dimension 35
        if self.dimension < 35:
            mu_block, sigma_block = fim_blocks(self.dimension)
        else:
            mu_block, sigma_block = njit(fim_blocks)(self.dimension)

        return scipy.linalg.block_diag(mu_block, sigma_block)

    def export_dict(self, variational_parameters):
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            export_dict (dictionary): Dict containing distribution information
        """
        mean, cov = self.reconstruct_distribution_parameters(variational_parameters)
        export_dict = {
            "type": "fullrank_Normal",
            "mean": mean,
            "covariance": cov,
            "variational_parameters": variational_parameters,
        }
        return export_dict

    def conduct_reparameterization(self, variational_parameters, n_samples):
        """Conduct a reparameterization.

        Args:
            variational_parameters (np.ndarray): Array with variational parameters
            n_samples (int): Number of samples for current batch

        Returns:
            samples_mat (np.ndarray): Array of actual samples from the variational
            distribution
        """
        standard_normal_sample_batch = np.random.normal(0, 1, size=(n_samples, self.dimension))
        mean, _, L = self.reconstruct_distribution_parameters(variational_parameters)
        samples_mat = mean + np.dot(L, standard_normal_sample_batch.T)

        return samples_mat.T, standard_normal_sample_batch

    def jacobi_variational_parameters_reparameterization(
        self, standard_normal_sample_batch, variational_parameters
    ):
        r"""Calculate the gradient of the reparameterization.

        Args:
            standard_normal_sample_batch (np.ndarray): Standard normal distributed sample
                                                    batch
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            jacobi_reparameterization_batch (np.ndarray): Tensor with Jacobi matrices for the
            reparameterization trick. The first dimension
            loops over the individual samples, the second
            dimension over variational parameters and the last
            dimension over the dimensions within one sample

        **Note:**
            We assume that *grad_reconstruct_params* is a row-vector containing the partial
            derivatives of the reconstruction mapping of the actual distribution parameters
            w.r.t. the variational parameters.

            The variable *jacobi_parameters* is the (n_parameters :math:`\times` dim_sample)
            Jacobi matrix of the reparameterization w.r.t. the distribution parameters,
            with differentiating after the distribution
            parameters in different rows and different output dimensions of the sample per
            column.
        """
        jacobi_reparameterization_lst = []
        grad_reconstruct_params = self._grad_reconstruct_distribution_parameters()
        for sample in standard_normal_sample_batch:
            jacobi_mean = np.eye(self.dimension)
            jacobi_cholesky = np.tile(sample, (variational_parameters.size - self.dimension, 1))
            jacobi_cholesky[0, -1] = 0
            jacobi_parameters = np.vstack((jacobi_mean, jacobi_cholesky))
            jacobi_reparameterization_lst.append(jacobi_parameters * grad_reconstruct_params.T)

        jacobi_reparameterization_batch = np.array(jacobi_reparameterization_lst)
        return jacobi_reparameterization_batch


class MixtureModel(VariationalDistribution):
    r"""Mixture model variational distribution class.

    Every component is a member of the same distribution family. Uses the parameterization:
    :math:`parameters=[\lambda_0,\lambda_1,...,\lambda_{C},\lambda_{weights}]`
    where :math:`C` is the number of components, :math:`\\lambda_i` are the variational parameters
    of the ith component and :math:`\\lambda_{weights}` parameters such that the component weights
    are obtained by:
    :math:`weight_i=\frac{exp(\lambda_{weights,i})}{\sum_{j=1}^{C}exp(\lambda_{weights,j})}`

    This allows the weight parameters :math:`\lambda_{weights}` to be unconstrained.

    Attributes:
        num_components (int): Number of mixture components.
        base_distribution: Variational distribution object for the components.
        n_parameters (int): Number of parameters used in the parameterization.
    """

    def __init__(self, base_distribution, dimension, num_components):
        """Initialize mixture model.

        Args:
            dimension (int): Dimension of the random variable
            num_components (int): Number of mixture components
            base_distribution: Variational distribution object for the components
        """
        super().__init__(dimension)
        self.num_components = num_components
        self.base_distribution = base_distribution
        self.n_parameters = num_components * base_distribution.n_parameters

    def initialize_variational_parameters(self, random=False):
        r"""Initialize variational parameters.

        Default weights initialization:
            :math:`w_i=\frac{1}{N_\text{sample space}}`

        Random weights intialization:
            :math:`w_i=\frac{s}{N_\text{experiments}}` where :math:`s` is a sample of a multinomial
            distribution with :math:`N_\text{experiments}`

        The component initialization is handle by the component itself.

        Args:
            random (bool, optional): If True, a random initialization is used. Otherwise the
                                     default is selected

        Returns:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
        """
        variational_parameters_components = (
            self.base_distribution.initialize_variational_parameters(random)
        )
        # Repeat for each component

        variational_parameters_components = np.tile(
            variational_parameters_components, self.num_components
        )
        if random:
            variational_parameters_weights = (
                np.random.multinomial(100, [1 / self.n_parameters] * self.n_parameters) / 100
            )
            variational_parameters_weights = np.log(variational_parameters_weights)
        else:
            variational_parameters_weights = np.log(np.ones(self.n_parameters) / self.n_parameters)

        return np.concatenate([variational_parameters_components, variational_parameters_weights])

    def construct_variational_parameters(
        self, component_parameters_list, weights
    ):  # pylint: disable=arguments-differ
        """Construct the variational parameters from the probabilities.

        Args:
            component_parameters_list (list): List of the component parameters of the components
            probabilities (np.ndarray): Probabilities of the distribution

        Returns:
            variational_parameters (np.ndarray): Variational parameters
        """
        variational_parameters = []
        for component_parameters in component_parameters_list:
            variational_parameters.append(
                self.base_distribution.construct_variational_parameters(*component_parameters)
            )
        variational_parameters.append(np.log(weights).flatten())
        return np.concatenate(variational_parameters)

    def _construct_component_variational_parameters(self, variational_parameters):
        """Reconstruct the weights and parameters of the mixture components.

        Creates a list containing the variational parameters of the different components.

        The list is nested, each entry correspond to the parameters of a component.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            variational_parameters_list (list): List of the variational parameters of the components
            weights (np.ndarray): Weights of the mixture
        """
        n_parameters_comp = self.base_distribution.n_parameters
        variational_parameters_list = []
        for j in range(self.num_components):
            params_comp = variational_parameters[
                n_parameters_comp * j : n_parameters_comp * (j + 1)
            ]
            variational_parameters_list.append(params_comp)
        # Compute the weights from the weight parameters
        weights = np.exp(variational_parameters[-self.num_components :])
        weights = weights / np.sum(weights)
        return variational_parameters_list, weights

    def reconstruct_distribution_parameters(self, variational_parameters):
        """Reconstruct the weights and parameters of the mixture components.

        The list is nested, each entry correspond to the parameters of a component.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            distribution_parameters_list (list): List of the distribution parameters of the
                                                 components
            weights (np.ndarray): Weights of the mixture
        """
        n_parameters_comp = self.base_distribution.n_parameters
        distribution_parameters_list = []
        for j in range(self.num_components):
            params_comp = variational_parameters[
                n_parameters_comp * j : n_parameters_comp * (j + 1)
            ]
            distribution_parameters_list.append(
                self.base_distribution.reconstruct_distribution_parameters(params_comp)
            )

        # Compute the weights from the weight parameters
        weights = np.exp(variational_parameters[-self.num_components :])
        weights = weights / np.sum(weights)
        return distribution_parameters_list, weights

    def draw(self, variational_parameters, num_draws=1):
        """Draw *num_draw* samples from the variational distribution.

        Uses a two-step process:
            1. From a multinomial distribution, based on the weights, select a component
            2. Sample from the selected component

        Args:
            variational_parameters (np.ndarray): Variational parameters
            num_draws (int): Number of samples to draw

        Returns:
            samples (np.ndarray): Row wise samples of the variational distribution
        """
        parameters_list, weights = self._construct_component_variational_parameters(
            variational_parameters
        )
        samples = []
        for _ in range(num_draws):
            # Select component to draw from
            component = np.argmax(np.random.multinomial(1, weights))
            # Draw a sample of this component
            sample = self.base_distribution.draw(parameters_list[component], 1)
            samples.append(sample)
        samples = np.concatenate(samples, axis=0)
        return samples

    def logpdf(self, variational_parameters, x):
        """Logpdf evaluated using the variational parameters at samples *x*.

        Is a general implementation using the logpdf function of the components. Uses the
        log-sum-exp trick [1] in order to reduce floating point issues.

        References:
        [1] :  David M. Blei, Alp Kucukelbir & Jon D. McAuliffe (2017) Variational Inference: A
               Review for Statisticians, Journal of the American Statistical Association, 112:518

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            logpdf (np.ndarray): Row vector of the logpdfs
        """
        parameters_list, weights = self._construct_component_variational_parameters(
            variational_parameters
        )
        logpdf = []
        x = np.atleast_2d(x)
        # Parameter for the log-sum-exp trick
        max_logpdf = -np.inf * np.ones(len(x))
        for j in range(self.num_components):
            logpdf.append(np.log(weights[j]) + self.base_distribution.logpdf(parameters_list[j], x))
            max_logpdf = np.maximum(max_logpdf, logpdf[-1])
        logpdf = np.array(logpdf) - np.tile(max_logpdf, (self.num_components, 1))
        logpdf = np.sum(np.exp(logpdf), axis=0)
        logpdf = np.log(logpdf) + max_logpdf
        return logpdf

    def pdf(self, variational_parameters, x):
        """Pdf evaluated using the variational parameters at given samples `x`.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            pdf (np.ndarray): Row vector of the pdfs
        """
        pdf = np.exp(self.logpdf(variational_parameters, x))
        return pdf

    def grad_params_logpdf(self, variational_parameters, x):
        """Logpdf gradient w.r.t. the variational parameters.

        Evaluated at samples *x*. Also known as the score function.
        Is a general implementation using the score functions of
        the components.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            score (np.ndarray): Column-wise scores
        """
        parameters_list, weights = self._construct_component_variational_parameters(
            variational_parameters
        )
        x = np.atleast_2d(x)
        # Jacobian of the weights w.r.t. weight parameters
        jacobian_weights = np.diag(weights) - np.outer(weights, weights)
        # Score function entries due to the parameters of the components
        component_block = []
        # Score function entries due to the weight parameterization
        weights_block = np.zeros((self.num_components, len(x)))
        logpdf = self.logpdf(variational_parameters, x)
        for j in range(self.num_components):
            # coefficient for the score term of every component
            precoeff = np.exp(self.base_distribution.logpdf(parameters_list[j], x) - logpdf)
            # Score function of the jth component
            score_comp = self.base_distribution.grad_params_logpdf(parameters_list[j], x)
            component_block.append(
                weights[j] * np.tile(precoeff, (len(score_comp), 1)) * score_comp
            )
            weights_block += np.tile(precoeff, (self.num_components, 1)) * jacobian_weights[
                :, j
            ].reshape(-1, 1)
        score = np.vstack((np.concatenate(component_block, axis=0), weights_block))
        return score

    def fisher_information_matrix(self, variational_parameters, n_samples=10000):
        """Approximate the Fisher information matrix using Monte Carlo.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            n_samples (int, optional): number of samples for a MC FIM estimation

        Returns:
            FIM (np.ndarray): Matrix (num parameters x num parameters)
        """
        samples = self.draw(variational_parameters, n_samples)
        scores = self.grad_params_logpdf(variational_parameters, samples)
        fim = 0
        for j in range(n_samples):
            fim = fim + np.outer(scores[:, j], scores[:, j])
        fim = fim / n_samples
        return fim

    def export_dict(self, variational_parameters):
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            export_dict (dictionary): Dict containing distribution information
        """
        parameters_list, weights = self._construct_component_variational_parameters(
            variational_parameters
        )
        export_dict = {
            "type": "mixture_model",
            "dimension": self.dimension,
            "num_components": self.num_components,
            "weights": weights,
            "variational_parameters": variational_parameters,
        }
        # Loop over the components
        for j in range(self.num_components):
            component_dict = self.base_distribution.export_dict(parameters_list[j])
            component_key = "component_" + str(j)
            export_dict.update({component_key: component_dict})
        return export_dict


class ParticleVariational(VariationalDistribution):
    r"""Variational distribution for particle distributions.

    The probabilities of the distribution are parameterized by softmax:
    :math:`p_i=p(\lambda_i)=\frac{\exp(\lambda_i)}{\sum_k exp(\lambda_k)}`

    Attributes:
        particles_obj (ParticleDiscreteDistribution): Particle distribution object
        dimension (int): Number of random variables
    """

    def __init__(self, dimension, sample_space):
        """Initialize variational distribution."""
        self.particles_obj = ParticleDiscreteDistribution(np.ones(len(sample_space)), sample_space)
        super().__init__(self.particles_obj.dimension)
        self.n_parameters = len(sample_space)

    def construct_variational_parameters(
        self, probabilities, sample_space
    ):  # pylint: disable=arguments-differ
        """Construct the variational parameters from the probabilities.

        Args:
            probabilities (np.ndarray): Probabilities of the distribution
            sample_space (np.ndarray): Sample space of the distribution

        Returns:
            variational_parameters (np.ndarray): Variational parameters
        """
        self.particles_obj = ParticleDiscreteDistribution(probabilities, sample_space)
        variational_parameters = np.log(probabilities).flatten()
        return variational_parameters

    def initialize_variational_parameters(self, random=False):
        r"""Initialize variational parameters.

        Default initialization:
            :math:`w_i=\frac{1}{N_\text{sample space}}`

        Random intialization:
            :math:`w_i=\frac{s}{N_\text{experiments}}` where :math:`s` is a sample of a multinomial
            distribution with :math:`N_\text{experiments}`

        Args:
            random (bool, optional): If True, a random initialization is used. Otherwise the
                                     default is selected

        Returns:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
        """
        if random:
            variational_parameters = (
                np.random.multinomial(100, [1 / self.n_parameters] * self.n_parameters) / 100
            )
            variational_parameters = np.log(variational_parameters)
        else:
            variational_parameters = np.log(np.ones(self.n_parameters) / self.n_parameters)

        return variational_parameters

    def reconstruct_distribution_parameters(self, variational_parameters):
        """Reconstruct probabilities from the variational parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            probabilities (np.ndarray): Probabilities of the distribution
        """
        probabilities = np.exp(variational_parameters)
        probabilities /= np.sum(probabilities)
        self.particles_obj = ParticleDiscreteDistribution(
            probabilities, self.particles_obj.sample_space
        )
        return probabilities, self.particles_obj.sample_space

    def draw(self, variational_parameters, num_draws=1):
        """Draw *num_draws* samples from distribution.

        Args:
            variational_parameters (np.ndarray): Variational parameters of the distribution
            num_draws (int): Number of samples

        Returns:
            samples (np.ndarray): samples (num_draws x n_dim)
        """
        self.reconstruct_distribution_parameters(variational_parameters)
        return self.particles_obj.draw(num_draws)

    def logpdf(self, variational_parameters, x):
        """Evaluate the natural logarithm of the logpdf at sample.

        Args:
            variational_parameters (np.ndarray): Variational parameters of the distribution
            x (np.ndarray): Locations at which to evaluate the distribution (n_samples x n_dim)

        Returns:
            logpdf (np.ndarray): Logpdfs at the locations x
        """
        self.reconstruct_distribution_parameters(variational_parameters)
        return self.particles_obj.logpdf(x)

    def pdf(self, variational_parameters, x):
        """Evaluate the probability density function (pdf) at sample.

        Args:
            variational_parameters (np.ndarray): Variational parameters of the distribution
            x (np.ndarray): Locations at which to evaluate the distribution (n_samples x n_dim)

        Returns:
            logpdf (np.ndarray): Pdfs at the locations x
        """
        self.reconstruct_distribution_parameters(variational_parameters)
        return self.particles_obj.pdf(x)

    def grad_params_logpdf(self, variational_parameters, x):
        r"""Logpdf gradient w.r.t. the variational parameters.

        Evaluated at samples  *x*. Also known as the score function.

        For the given parameterization, the score function yields:
        :math:`\nabla_{\lambda_i}\ln p(\theta_j | \lambda)=\delta_{ij}-p_i`

        Args:
            variational_parameters (np.ndarray): Variational parameters of the distribution
            x (np.ndarray): Locations at which to evaluate the distribution (n_samples x n_dim)

        Returns:
            score_function (np.ndarray): Score functions at the locations x
        """
        self.reconstruct_distribution_parameters(variational_parameters)
        index = np.array(
            [(self.particles_obj.sample_space == xi).all(axis=1).nonzero()[0] for xi in x]
        ).flatten()

        if len(index) != len(x):
            raise ValueError(
                f"At least one event is not part of the sample space "
                f"{self.particles_obj.sample_space}"
            )
        sample_scores = np.eye(len(variational_parameters)) - np.exp(
            variational_parameters
        ) / np.sum(np.exp(variational_parameters))
        # Get the samples
        return sample_scores[index].T

    def fisher_information_matrix(self, variational_parameters):
        r"""Compute the fisher information matrix.

        For the given parameterization, the Fisher information yields:
        :math:`\text{FIM}_{ij}=\delta_{ij} p_i -p_i p_j`

        Args:
            variational_parameters (np.ndarray): Variational parameters of the distribution

        Returns:
            fim (np.ndarray): Fisher information matrix (n_params x n_params)
        """
        probabilities, _ = self.reconstruct_distribution_parameters(variational_parameters)
        fim = np.diag(probabilities) - np.outer(probabilities, probabilities)
        return fim

    def export_dict(self, variational_parameters):
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            export_dict (dictionary): Dict containing distribution information
        """
        self.reconstruct_distribution_parameters(variational_parameters)
        export_dict = {
            "type": type(self),
            "variational_parameters": variational_parameters,
        }
        export_dict.update(self.particles_obj.export_dict())
        return export_dict


def create_simple_distribution(distribution_options):
    """Create a simple variational distribution object.

    No nested distributions like mixture models.

    Args:
        distribution_options (dict): Dict for the distribution options

    Returns:
        distribution_obj: Variational distribution object
    """
    distribution_family = distribution_options.get('variational_family', None)
    if distribution_family == "normal":
        dimension = distribution_options.get('dimension')
        approximation_type = distribution_options.get('variational_approximation_type', None)
        distribution_obj = create_normal_distribution(dimension, approximation_type)
    elif distribution_family == "particles":
        dimension = distribution_options["dimension"]
        probabilities = distribution_options["probabilities"]
        sample_space = distribution_options["sample_space"]
        distribution_obj = ParticleDiscreteDistribution(probabilities, sample_space)

    return distribution_obj


def create_normal_distribution(dimension, approximation_type):
    """Create a normal variational distribution object.

    Args:
        dimension (int): Dimension of latent variable
        approximation type (str): Full rank or mean field

    Returns:
        distribution_obj: Variational distribution object
    """
    if approximation_type == "mean_field":
        distribution_obj = MeanFieldNormalVariational(dimension)
    elif approximation_type == "fullrank":
        distribution_obj = FullRankNormalVariational(dimension)
    else:
        supported_types = {'mean_field', 'fullrank'}
        raise ValueError(
            f"Requested variational approximation type not supported: {approximation_type}.\n"
            f"Supported types:  {supported_types}. "
        )
    return distribution_obj


def create_mixture_model_distribution(base_distribution, dimension, num_components):
    """Create a mixture model variational distribution.

    Args:
        base_distribution: Variational distribution object
        dimension (int): Dimension of latent variable
        num_components (int): Number of mixture components

    Returns:
        distribution_obj: Variational distribution object
    """
    if num_components > 1:
        distribution_obj = MixtureModel(base_distribution, dimension, num_components)
    else:
        raise ValueError(
            "Number of mixture components has be greater than 1. If a single component is"
            "desired use the respective variational distribution directly (is computationally"
            "more efficient)."
        )
    return distribution_obj


def create_variational_distribution(distribution_options):
    """Create variational distribution object from dictionary.

    Args:
        distribution_options (dict): Dictionary containing parameters
                                     defining the distribution

    Returns:
        distribution: Variational distribution object
    """
    distribution_family = distribution_options.get('variational_family', None)
    supported_simple_distribution_families = ['normal', 'particles']
    supported_nested_distribution_families = ['mixture_model']
    if distribution_family in supported_simple_distribution_families:
        distribution_obj = create_simple_distribution(distribution_options)
    elif distribution_family in supported_nested_distribution_families:
        dimension = distribution_options.get('dimension')
        num_components = distribution_options.get('num_components')
        base_distribution_options = distribution_options.get('base_distribution')
        base_distribution_options.update({"dimension": dimension})
        base_distribution_obj = create_simple_distribution(base_distribution_options)
        distribution_obj = create_mixture_model_distribution(
            base_distribution_obj, dimension, num_components
        )
    else:
        supported_distributions = (
            supported_nested_distribution_families + supported_simple_distribution_families
        )
        raise ValueError(
            f"Requested variational family type not supported: {distribution_family}.\n"
            f"Supported types:  {supported_distributions}."
        )
    return distribution_obj
