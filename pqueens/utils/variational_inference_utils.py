"""Variational distribution utils."""

import abc

import numpy as np
import scipy
from numba import njit


class VariationalDistribution:
    """Base class for probability distributions for variational inference."""

    def __init__(self, dimension):
        """Initialize variational distribution."""
        self.dimension = dimension

    @abc.abstractmethod
    def draw(self, variational_params, num_draws=1):
        """Draw num_draws samples from distribution."""

    @abc.abstractmethod
    def logpdf(self, variational_params, x):
        """Evaluate the natural logarithm of the logpdf at sample."""

    @abc.abstractmethod
    def pdf(self, variational_params, x):
        """Evaluate the probability density function (pdf) at sample."""

    @abc.abstractmethod
    def grad_params_logpdf(self, variational_params, x):
        """Logpdf gradient w.r.t. to the variational parameters.

        Evaluated at samples x. Also known as the score function.
        """

    @abc.abstractmethod
    def fisher_information_matrix(self, variational_params, num_samples):
        """Compute the fisher information matrix.

        Depends on the variational distribution for the given
        parameterization.
        """

    def conduct_reparameterization(self):
        """Conduct a reparameterization."""
        raise NotImplementedError(
            "The reparameterization trick is not available for the variational distribution "
            f"family {self.__class__.__name__}. Abort..."
        )

    def jacobi_variational_params_reparameterization(self):
        """Calculate the gradient of the reparameterization."""
        raise NotImplementedError(
            "The gradient of the reparameterization is not available for the variational "
            f"distribution family {self.__class__.__name__}. Abort..."
        )


class MeanFieldNormalVariational(VariationalDistribution):
    r"""Mean field multivariate normal distribution.

     Uses the parameterization (as in [1])
    :math:`parameters=[\mu, \lambda]` where :math:`mu` are the mean values and
    :math:`\sigma^2=exp(2*\lambda)` the variances allowing for :math:`\lambda` to be
    unconstrained.

    References:
        [1]: Kucukelbir, Alp, et al. "Automatic differentiation variational inference."
             The Journal of Machine Learning Research 18.1 (2017): 430-474.

    Attributes:
        dimension (int): Dimension of the random variable
        num_params (int): Number of parameters used in the parameterization
    """

    def __init__(self, dimension):
        """Initialize variational distribution.

        Args:
            dimension (int): Dimension of RV.
        """
        super().__init__(dimension)
        self.num_params = 2 * dimension

    def initialize_parameters_randomly(self):
        r"""Initialize the variational parameters randomly.

        Based on
        :math:`\mu=Uniform(-0.1,0.1)`
        :math:`\sigma^2=Uniform(0.9,1.1)`

        Returns:
            variational_params (np.array): Variational parameters
        """
        variational_params = np.hstack(
            (
                0.1 * (-0.5 + np.random.rand(self.dimension)),
                0.5 + np.log(1 + 0.1 * (-0.5 + np.random.rand(self.dimension))),
            )
        )
        return variational_params

    @staticmethod
    def construct_variational_params(mean, covariance):
        """Construct the variational parameters from mean and covariance.

        Args:
            mean (np.array): Mean values of the distribution
            covariance (np.array): Covariance matrix of the distribution

        Returns:
            variational_params (np.array): Variational parameters
        """
        if len(mean) == len(covariance):
            variational_params = np.hstack((mean.flatten(), 0.5 * np.log(np.diag(covariance))))
        else:
            raise ValueError(
                f"Dimension of the mean value {len(mean)} does not equal covariance dimension"
                f"{covariance.shape}"
            )
        return variational_params

    def reconstruct_parameters(self, variational_params):
        """Reconstruct mean and covariance from the variational parameters.

        Args:
            variational_params (np.array): Variational parameters

        Returns:
            mean (np.array): Mean value of the distribution
            cov (np.array): Covariance of the distribution
        """
        mean, cov = (
            variational_params[: self.dimension],
            np.exp(2 * variational_params[self.dimension :]),
        )
        return mean, cov

    def _grad_reconstruct_parameters(self, variational_params):
        """Gradient of the parameter reconstruction.

         Args:
            variational_params (np.array): Variational parameters

        Returns:
            grad_reconstruct_params (np.array): Gradient vector of the reconstruction
                                                w.r.t. the variational parameters
        """
        grad_mean = np.ones((1, self.dimension))
        grad_cov = (np.exp(2 * variational_params[self.dimension :]) * 2).reshape(1, -1)
        grad_reconstruct_params = np.hstack((grad_mean, grad_cov))
        return grad_reconstruct_params

    def draw(self, variational_params, num_draws=1):
        """Draw `num_draw` samples from the variational distribution.

        Args:
            variational_params (np.array): Variational parameters
            num_draw (int): Number of samples to draw

        Returns:
            samples (np.array): Row-wise samples of the variational distribution
        """
        mean, cov = self.reconstruct_parameters(variational_params)
        samples = np.random.randn(num_draws, self.dimension) * np.sqrt(cov).reshape(
            1, -1
        ) + mean.reshape(1, -1)
        return samples

    def logpdf(self, variational_params, x):
        """Logpdf evaluated using the variational parameters at samples `x`.

        Args:
            variational_params (np.array): Variational parameters
            x (np.array): Row-wise samples

        Returns:
            logpdf (np.array): Rowvector of the logpdfs
        """
        mean, cov = self.reconstruct_parameters(variational_params)
        x = np.atleast_2d(x)
        logpdf = (
            -0.5 * self.dimension * np.log(2 * np.pi)
            - np.sum(variational_params[self.dimension :])
            - 0.5 * np.sum((x - mean) ** 2 / cov, axis=1)
        )
        return logpdf.flatten()

    def pdf(self, variational_params, x):
        """Pdf of the variational distribution evaluated at samples `x`.

        First computes the logpdf, which numerically more stable for exponential distributions.

        Args:
            variational_params (np.array): Variational parameters
            x (np.array): Row-wise samples

        Returns:
            pdf (np.array): Rowvector of the pdfs
        """
        pdf = np.exp(self.logpdf(variational_params, x))
        return pdf

    def grad_params_logpdf(self, variational_params, x):
        """Logpdf gradient w.r.t. to the variational parameters.

        Evaluated at samples x. Also known as the score function.

        Args:
            variational_params (np.array): Variational parameters
            x (np.array): Row-wise samples

        Returns:
            score (np.array): Column-wise scores
        """
        mean, cov = self.reconstruct_parameters(variational_params)
        dlnN_dmu = (x - mean) / cov
        dlnN_dsigma = (x - mean) ** 2 / cov - np.ones(x.shape)
        score = np.concatenate(
            [
                dlnN_dmu.T.reshape(self.dimension, len(x)),
                dlnN_dsigma.T.reshape(self.dimension, len(x)),
            ]
        )
        return score

    def grad_logpdf_sample(self, sample_batch, variational_params):
        """Computes the gradient of the logpdf w.r.t. to the x.

        Args:
            variational_params (np.array): Variational parameters
            sample_batch (np.array): Row-wise samples

        Returns:
            gradients_batch (np.array): Gradients of the log-pdf w.r.t. the
                                        sample x. The first dimension of the
                                        array corresponds to the different samples.
                                        The second dimension to different dimensions
                                        within one sample. (Third dimension is empty
                                        and just added to keep slices two dimensional.)
        """
        mean, cov = self.reconstruct_parameters(variational_params)
        gradient_lst = []
        for sample in sample_batch:
            gradient_lst.append((-2 * (sample - mean) / cov).reshape(-1, 1))

        gradients_batch = np.array(gradient_lst)
        return gradients_batch

    def fisher_information_matrix(self, variational_params):
        """Compute the Fisher information matrix analytically.

        Args:
            variational_params (np.array): Variational parameters

        Returns:
            FIM (np.array): Matrix (num parameters x num parameters)
        """
        fisher_diag = np.exp(-2 * variational_params[self.dimension :])
        fisher_diag = np.hstack((fisher_diag, 2 * np.ones(self.dimension)))
        return np.diag(fisher_diag)

    def export_dict(self, variational_params):
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_params (np.array): Variational parameters

        Returns:
            export_dict (dictionary): Dict containing distribution information
        """
        mean, cov = self.reconstruct_parameters(variational_params)
        sd = cov**0.5
        export_dict = {
            "type": "meanfield_Normal",
            "mean": mean,
            "covariance": np.diag(cov),
            "standard_deviation": sd,
            "variational_parameters": variational_params,
        }
        return export_dict

    def conduct_reparameterization(self, variational_params, n_samples):
        """Conduct a reparameterization.

        Args:
            variational_params (np.array): Array with variational parameters
            n_samples (int): Number of samples for current batch

        Returns:
            samples_mat (np.array): Array of actual samples from the variational
                                   distribution
            standard_normal_sample_batch (np.array): Standard normal distributed sample
                                                     batch
        """
        standard_normal_sample_batch = np.random.normal(0, 1, size=(n_samples, self.dimension))
        mean, cov = self.reconstruct_parameters(variational_params)
        samples_mat = mean + np.sqrt(cov) * standard_normal_sample_batch

        return samples_mat, standard_normal_sample_batch

    def jacobi_variational_params_reparameterization(
        self, standard_normal_sample_batch, variational_params
    ):
        """Calculate the gradient of the reparameterization.

        Args:
            standard_normal_sample_batch (np.array): Standard normal distributed sample
                                                    batch
            variational_params (np.array): Variational parameters

        Note:
            We assume that `grad_reconstruct_params` is a row-vector containing the partial
            derivatives of the reconstruction mapping of the actual distribution parameters
            w.r.t. the variational parameters.

            The variable `jacobi_parameters` is the (num_params x dim_sample) Jacobi matrix
            of the reparameterization w.r.t. the distribution parameters,
            with differentiating after the distribution
            parameters in different rows and different output dimensions of the sample per
            column.

        Returns:
            jacobi_reparameterization_batch (np.array): Tensor with Jacobi matrices for the
                                                  reparameterization trick. The first dimension
                                                  loops over the individual samples, the second
                                                  dimension over variational parameters and the last
                                                  dimension over the dimensions within one sample
        """
        jacobi_reparameterization_lst = []
        grad_reconstruct_params = self._grad_reconstruct_parameters(variational_params)
        for sample in standard_normal_sample_batch:
            jacobi_parameters = np.vstack((np.eye(self.dimension), np.diag(sample)))
            jacobi_reparameterization = jacobi_parameters * grad_reconstruct_params.T
            jacobi_reparameterization_lst.append(jacobi_reparameterization)

        jacobi_reparameterization_batch = np.array(jacobi_reparameterization_lst)
        return jacobi_reparameterization_batch


class FullRankNormalVariational(VariationalDistribution):
    r"""Fullrank multivariate normal distribution.

    Uses the parameterization (as in [1])
    :math:`parameters=[\mu, \lambda]` where :math:`\mu` are the mean values and
    :math:`\lambda` is an array containing the nonzero entries of the lower Cholesky
    decomposition of the covariance matrix :math:`L`:
    :math:`\lambda=[L_{00},L_{10},L_{11},L_{20},L_{21},L_{22}, ...]`.
    This allows the parameters :math:`\lambda` to be unconstrained.

    References:
        [1]: Kucukelbir, Alp, et al. "Automatic differentiation variational inference."
             The Journal of Machine Learning Research 18.1 (2017): 430-474.

    Attributes:
        dimension (int): Dimension of the random variable
        num_params (int): Number of parameters used in the parameterization
    """

    def __init__(self, dimension):
        """Initialize variational distribution.

        Args:
            dimension (int): dimension of the RV
        """
        super().__init__(dimension)
        self.num_params = (dimension * (dimension + 1)) // 2 + dimension

    def initialize_parameters_randomly(self):
        r"""Initialize the variational parameters randomly.

        By
        :math:`\mu=Uniform(-0.1,0.1)`
        :math:`L=diag(Uniform(0.9,1.1))` where :math:`\Sigma=LL^T`

        Returns:
            variational_params (np.array): Variational parameters
        """
        cholesky_covariance = np.eye(self.dimension) + 0.1 * (
            -0.5 + np.diag(np.random.rand(self.dimension))
        )
        variational_params = np.zeros(self.dimension) + 0.1 * (
            -0.5 + np.random.rand(self.dimension)
        )
        for j in range(len(cholesky_covariance)):
            variational_params = np.hstack((variational_params, cholesky_covariance[j, : j + 1]))
        return variational_params

    @staticmethod
    def construct_variational_params(mean, covariance):
        """Construct the variational parameters from mean and covariance.

        Args:
            mean (np.array): Mean values of the distribution
            covariance (np.array): Covariance matrix of the distribution

        Returns:
            variational_params (np.array): Variational parameters
        """
        if len(mean) == len(covariance):
            cholesky_covariance = np.linalg.cholesky(covariance)
            variational_params = mean.flatten()
            for j in range(len(cholesky_covariance)):
                variational_params = np.hstack(
                    (variational_params, cholesky_covariance[j, : j + 1])
                )
        else:
            raise ValueError(
                f"Dimension of the mean value {len(mean)} does not equal covariance dimension"
                f"{covariance.shape}"
            )
        return variational_params

    def reconstruct_parameters(self, variational_params):
        """Reconstruct mean value, covariance and its Cholesky decomposition.

        Args:
            variational_params (np.array): Variational parameters

        Returns:
            mean (np.array): Mean value of the distribution
            cov (np.array): Covariance of the distribution
            L (np.array): Cholesky decomposition of the covariance matrix of the distribution
        """
        mean = variational_params[: self.dimension].reshape(-1, 1)
        cholesky_covariance_array = variational_params[self.dimension :]
        cholesky_covariance = np.zeros((self.dimension, self.dimension))
        idx = np.tril_indices(self.dimension, k=0, m=self.dimension)
        cholesky_covariance[idx] = cholesky_covariance_array
        cov = np.matmul(cholesky_covariance, cholesky_covariance.T)
        return mean, cov, cholesky_covariance

    def _grad_reconstruct_parameters(self, variational_params):
        """Gradient of the parameter reconstruction.

         Args:
            variational_params (np.array): Variational parameters

        Returns:
            grad_reconstruct_params (np.array): Gradient vector of the reconstruction
                                                w.r.t. the variational parameters
        """
        grad_mean = np.ones((1, self.dimension))
        grad_cholesky = np.ones((1, variational_params.size - self.dimension))
        grad_reconstruct_params = np.hstack((grad_mean, grad_cholesky))
        return grad_reconstruct_params

    def draw(self, variational_params, num_draws=1):
        """Draw `num_draw` samples from the variational distribution.

        Args:
            variational_params (np.array): Variational parameters
            num_draw (int): Number of samples to draw

        Returns:
            samples (np.array): Row-wise samples of the variational distribution
        """
        mean, _, L = self.reconstruct_parameters(variational_params)
        sample = np.dot(L, np.random.randn(self.dimension, num_draws)).T + mean.reshape(1, -1)
        return sample

    def logpdf(self, variational_params, x):
        """Logpdf evaluated using the at samples `x`.

        Args:
            variational_params (np.array): Variational parameters
            x (np.array): Row-wise samples

        Returns:
            logpdf (np.array): Rowvector of the logpdfs
        """
        mean, cov, L = self.reconstruct_parameters(variational_params)
        x = np.atleast_2d(x)
        u = np.linalg.solve(cov, (x.T - mean))
        col_dot_prod = lambda x, y: np.sum(x * y, axis=0)
        logpdf = (
            -0.5 * self.dimension * np.log(2 * np.pi)
            - np.sum(np.log(np.abs(np.diag(L))))
            - 0.5 * col_dot_prod(x.T - mean, u)
        )
        return logpdf.flatten()

    def pdf(self, variational_params, x):
        """Pdf of evaluted at given samples `x`.

        First computes the logpdf, which numerically more stable for exponential distributions.

        Args:
            variational_params (np.array): Variational parameters
            x (np.array): Row-wise samples

        Returns:
            pdf (np.array): Rowvector of the pdfs
        """
        pdf = np.exp(self.logpdf(variational_params, x))
        return pdf

    def grad_params_logpdf(self, variational_params, x):
        """Logpdf gradient w.r.t. to the variational parameters.

        Evaluated at samples x. Also known as the score function.

        Args:
            variational_params (np.array): Variational parameters
            x (np.array): Row-wise samples

        Returns:
            score (np.array): Column-wise scores
        """
        mean, cov, L = self.reconstruct_parameters(variational_params)
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

    def grad_logpdf_sample(self, sample_batch, variational_params):
        """Computes the gradient of the logpdf w.r.t. to the x.

        Args:
            variational_params (np.array): Variational parameters
            sample_batch (np.array): Row-wise samples

        Returns:
            gradients_batch (np.array): Gradients of the log-pdf w.r.t. the
                                        sample x. The first dimension of the
                                        array corresponds to the different samples.
                                        The second dimension to different dimensions
                                        within one sample. (Third dimension is empty
                                        and just added to keep slices two dimensional.)
        """
        mean, cov, _ = self.reconstruct_parameters(variational_params)
        gradient_lst = []
        for sample in sample_batch:
            gradient_lst.append(
                np.dot(np.linalg.inv(cov), 2 * (sample.reshape(-1, 1) - mean)).reshape(-1, 1)
            )

        gradients_batch = np.array(gradient_lst)
        return gradients_batch

    def fisher_information_matrix(self, variational_params):
        """Compute the Fisher information matrix analytically.

        Args:
            variational_params (np.array): Variational parameters

        Returns:
            FIM (np.array): Matrix (num parameters x num parameters)
        """
        _, cov, L = self.reconstruct_parameters(variational_params)

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

    def export_dict(self, variational_params):
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_params (np.array): Variational parameters

        Returns:
            export_dict (dictionary): Dict containing distribution information
        """
        mean, cov, _ = self.reconstruct_parameters(variational_params)
        export_dict = {
            "type": "fullrank_Normal",
            "mean": mean,
            "covariance": cov,
            "variational_parameters": variational_params,
        }
        return export_dict

    def conduct_reparameterization(self, variational_params, n_samples):
        """Conduct a reparameterization.

        Args:
            variational_params (np.array): Array with variational parameters
            n_samples (int): Number of samples for current batch

        Returns:
            samples_mat (np.array): Array of actual samples from the variational
                                   distribution
        """
        standard_normal_sample_batch = np.random.normal(0, 1, size=(n_samples, self.dimension))
        mean, _, L = self.reconstruct_parameters(variational_params)
        samples_mat = mean + np.dot(L, standard_normal_sample_batch.T)

        return samples_mat.T, standard_normal_sample_batch

    def jacobi_variational_params_reparameterization(
        self, standard_normal_sample_batch, variational_params
    ):
        """Calculate the gradient of the reparameterization.

        Args:
            standard_normal_sample_batch (np.array): Standard normal distributed sample
                                                    batch
            variational_params (np.array): Variational parameters

        Note:
            We assume that `grad_reconstruct_params` is a row-vector containing the partial
            derivatives of the reconstruction mapping of the actual distribution parameters
            w.r.t. the variational parameters.

            The variable `jacobi_parameters` is the (num_params x dim_sample) Jacobi matrix
            of the reparameterization w.r.t. the distribution parameters,
            with differentiating after the distribution
            parameters in different rows and different output dimensions of the sample per
            column.

        Returns:
            jacobi_reparameterization_batch (np.array): Tensor with Jacobi matrices for the
                                                  reparameterization trick. The first dimension
                                                  loops over the individual samples, the second
                                                  dimension over variational parameters and the last
                                                  dimension over the dimensions within one sample
        """
        jacobi_reparameterization_lst = []
        grad_reconstruct_params = self._grad_reconstruct_parameters(variational_params)
        for sample in standard_normal_sample_batch:
            jacobi_mean = np.eye(self.dimension)
            jacobi_cholesky = np.tile(sample, (variational_params.size - self.dimension, 1))
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
    are obtain by:
    :math:`weight_i=\frac{exp(\lambda_{weights,i})}{\sum_{j=1}^{C}exp(\lambda_{weights,j})}`

    This allows the weight parameters :math:`\lambda_{weights}` to be unconstrained.

    Attributes:
        dimension (int): Dimension of the random variable
        num_params (int): Number of parameters used in the parameterization
        num_components (int): Number of mixture components
        base_distribution: Variational distribution object for the components
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
        self.num_params = num_components * base_distribution.num_params

    def initialize_parameters_randomly(self):
        """Initialize the variational parameters.

        The weight parameters are
        initialized in a random (is said to be beneficial for the optimization)
        but bounded way such that no component has a dominating or extremely
        small weight in the beginning of the optimization. The parameters of the
        base distribution are initialized by the object itself.

        Args:
            None

        Returns:
            variational_params (np.array): Variational parameters
        """
        variational_params = []
        # Initialize the variational parameters of the components
        for _ in range(self.num_components):
            params_comp = self.base_distribution.initialize_parameters_randomly().tolist()
            variational_params.extend(params_comp)
        # Initialize weight parameters with random uniform noise
        params_weights = 1 + 0.1 * (np.random.rand(self.num_components) - 0.5)
        variational_params.extend(params_weights.tolist())
        variational_params = np.array(variational_params)
        return variational_params

    def reconstruct_parameters(self, variational_params):
        """Reconstruct the weights and parameters of the mixture components.

        Creates a list containing the variational parameters of the different components.

        Args:
            variational_params (np.array): Variational parameters

        Returns:
            variational_params_list (list): List of the variational parameters (np.array) of
                                            the different components.
            weights (np.array): Weights of the mixture
        """
        num_params_comp = self.base_distribution.num_params
        variational_params_list = []
        for j in range(self.num_components):
            params_comp = variational_params[num_params_comp * j : num_params_comp * (j + 1)]
            variational_params_list.append(params_comp)
        # Compute the weights from the weight parameters
        weights = np.exp(variational_params[-self.num_components :])
        weights = weights / np.sum(weights)
        return variational_params_list, weights

    def draw(self, variational_params, num_draws=1):
        """Draw `num_draw` samples from the variational distribution.

        Uses a two step process:
            1. From a multi-nomial distribution, based on the weights, select a component
            2. Sample from the selected component

        Args:
            variational_params (np.array): Variational parameters
            num_draw (int): Number of samples to draw

        Returns:
            samples (np.array): Row-wise samples of the variational distribution
        """
        parameters_list, weights = self.reconstruct_parameters(variational_params)
        samples = []
        for _ in range(num_draws):
            # Select component to draw from
            component = np.argmax(np.random.multinomial(1, weights))
            # Draw a sample of this component
            sample = self.base_distribution.draw(parameters_list[component], 1)
            samples.append(sample)
        samples = np.concatenate(samples, axis=0)
        return samples

    def logpdf(self, variational_params, x):
        """Logpdf evaluated using the variational parameters at samples `x`.

        Is a general implementation using the logpdf function of the components. Uses the
        log-sum-exp trick [1] in order to reduce floating point issues.

        [1] :  David M. Blei, Alp Kucukelbir & Jon D. McAuliffe (2017) Variational Inference: A
        Review for Statisticians, Journal of the American Statistical Association, 112:518

        Args:
            variational_params (np.array): Variational parameters
            x (np.array): Row-wise samples

        Returns:
            logpdf (np.array): Rowvector of the logpdfs
        """
        parameters_list, weights = self.reconstruct_parameters(variational_params)
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

    def pdf(self, variational_params, x):
        """Pdf evaluated using the variational parameters at given samples `x`.

        Args:
            variational_params (np.array): Variational parameters
            x (np.array): Row-wise samples

        Returns:
            pdf (np.array): Rowvector of the pdfs
        """
        pdf = np.exp(self.logpdf(variational_params, x))
        return pdf

    def grad_params_logpdf(self, variational_params, x):
        """Logpdf gradient w.r.t. to the variational parameters.

        Evaluated at samples x. Also known as the score function. Is a general implementation using
        the score functions of the components.

        Args:
            variational_params (np.array): Variational parameters
            x (np.array): Row-wise samples

        Returns:
            score (np.array): Column-wise scores
        """
        parameters_list, weights = self.reconstruct_parameters(variational_params)
        x = np.atleast_2d(x)
        # Jacobian of the weights w.r.t. weight parameters
        jacobian_weights = np.diag(weights) - np.outer(weights, weights)
        # Score function entries due to the parameters of the components
        component_block = []
        # Score function entries due to the weight parameterization
        weights_block = np.zeros((self.num_components, len(x)))
        logpdf = self.logpdf(variational_params, x)
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

    def fisher_information_matrix(self, variational_params, num_samples=1000):
        """Approximate the Fisher information matrix using Monte Carlo.

        Args:
            variational_params (np.array): Variational parameters
            num_samples (int): Number of samples used in the Monte Carlo estimation

        Returns:
            FIM (np.array): Matrix (num parameters x num parameters)
        """
        samples = self.draw(variational_params, num_samples)
        scores = self.grad_params_logpdf(variational_params, samples)
        fim = 0
        for j in range(num_samples):
            fim = fim + np.outer(scores[:, j], scores[:, j])
        fim = fim / num_samples
        return fim

    def export_dict(self, variational_params):
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_params (np.array): Variational parameters

        Returns:
            export_dict (dictionary): Dict containing distribution information
        """
        parameters_list, weights = self.reconstruct_parameters(variational_params)
        export_dict = {
            "type": "mixture_model",
            "dimension": self.dimension,
            "num_components": self.num_components,
            "weights": weights,
            "variational_parameters": variational_params,
        }
        # Loop over the components
        for j in range(self.num_components):
            component_dict = self.base_distribution.export_dict(parameters_list[j])
            component_key = "component_" + str(j)
            export_dict.update({component_key: component_dict})
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
    return distribution_obj


def create_normal_distribution(dimension, approximation_type):
    """Create a normal variational distribution object.

    Args:
        dimension (int): Dimension of latent variable
        approximation type (str): fullrank or mean field

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
        dimension (int): Dimension of latent variable
        num_components (int): Number of mixture components
        base_distribution: Variational distribution object

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
    supported_simple_distribution_families = ['normal']
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


def calculate_grad_log_variational_distr_variational_params(
    jacobi_reparameterization_variational_params_lst, grad_log_variational_distr_params_lst
):
    """Calculate the gradient of the log-variational distribution.

    w.r.t. variational parameters, evaluated at the current value of the variational
    params.

    Args:
        jacobi_reparameterization_variational_params_lst (lst): List of Jacobi matrices of the
                                                                reparameterization w.r.t. the
                                                                variational parameters.
        grad_log_variational_distr_params_lst (lst): List of gradients of the variational
                                                     distribution w.r.t. to the input
                                                     parameters per sample

    Returns:
        grad_log_variational_distr_variational_params_lst (np.array): List of gradients of the
                                                                  log-variational
                                                                  distribution w.r.t. variational
                                                                  parameters
    """
    # pylint: disable=line-too-long
    grad_log_variational_distr_variational_params_lst = []
    for (jacobi_reparameterization_variational_params, grad_log_variational_distr_params) in zip(
        jacobi_reparameterization_variational_params_lst, grad_log_variational_distr_params_lst
    ):
        grad_log_variational_distr_variational_params_lst.append(
            np.dot(jacobi_reparameterization_variational_params, grad_log_variational_distr_params)
        )
    # pyplint: enable=line-too-long

    return grad_log_variational_distr_variational_params_lst
