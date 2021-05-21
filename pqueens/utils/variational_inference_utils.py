import numpy as np
import scipy
import abc


class VariationalDistribution:
    """ Base class for probability distributions for variational inference."""

    def __init__(self, dimension):
        self.dimension = dimension

    @abc.abstractmethod
    def draw(self, variational_params, num_draws=1):
        """ 
        Draw num_draws samples from distribution.
        """
        pass

    @abc.abstractmethod
    def logpdf(self, variational_params, x):
        """
        Evaluate the natural logarithm of the logpdf at sample.
        """
        pass

    @abc.abstractmethod
    def pdf(self, variational_params, x):
        """
        Evaluate the probability density function (pdf) at sample.
        """
        pass

    @abc.abstractmethod
    def grad_params_logpdf(self, variational_params, x):
        """
        Evaluate the gradient of the logpdf w.r.t. variational params at the given samples. Also
        known as the score function of the distribution.
        """
        pass

    @abc.abstractmethod
    def fisher_information_matrix(self, variational_params, x):
        """
        Compute the fisher information matrix of the variational distribution for the given 
        parameterization.
        """
        pass


class MeanFieldNormalVariational(VariationalDistribution):
    """
    Uncorrelated mean field multivariate normal distribution. Uses the parameterization (as in [1])
    :math:`parameters=[\\mu, \\lambda]` where :math:`mu` are the mean values and 
    :math:`\\sigma^2=exp(2*\\lambda)` the variances allowing for :math:`\\lambda` to be 
    unconstrained. 
    
    References:
        [1]: Kucukelbir, Alp, et al. "Automatic differentiation variational inference."
             The Journal of Machine Learning Research 18.1 (2017): 430-474.

    Attributes:
        dimension (int): Dimension of the random variable
        num_params (int): Number of parameters used in the parameterization

    """

    def __init__(self, dimension):
        super(MeanFieldNormalVariational, self).__init__(dimension)
        self.num_params = 2 * dimension

    def initialize_parameters_randomly(self):
        """
        Initialize the variational parameters by 
        :math:`\\mu=Uniform(-0.1,0.1)`
        :math:`\\sigma^2=Uniform(0.9,1.1)`
        
        Args:
            None

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
        """
        Construct the variational parameters for a given mean vector and covariance matrix  

        Args:
            mean (np.array): Mean values of the distribution
            covariance (np.array): Covariance of the distribution

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
        """
        Reconstruct mean value and covariance of the distribution based on the variational 
        parameters

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

    def draw(self, variational_params, num_draws=1):
        """
        Draw `num_draw` samples from the variational distribution given variational parameters. 

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
        """
        Logpdf of the variational distribution evaluted using the variational parameters at given
         samples `x`.
        
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
        """
        Pdf of the variational distribution evaluted using the variational parameters at given 
        samples `x`. First computes the logpdf, which numerically more stable for exponential 
        distributions.
        
        Args: 
            variational_params (np.array): Variational parameters 
            x (np.array): Row-wise samples

        Returns:
            pdf (np.array): Rowvector of the pdfs

        """
        pdf = np.exp(self.logpdf(variational_params, x))
        return pdf

    def grad_params_logpdf(self, variational_params, x):
        """
        Computes the gradient of the logpdf w.r.t. to the variational parameters of the 
        distribution evaluated at samples x. Also known as the score function.

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

    def fisher_information_matrix(self, variational_params):
        """
        Compute the Fisher information matrix analytically.

        Args:
            variational_params (np.array): Variational parameters

        Returns: 
            FIM (np.array): Matrix (num parameters x num parameters)

        """
        fisher_diag = np.exp(-2 * variational_params[self.dimension :])
        fisher_diag = np.hstack((fisher_diag, 2 * np.ones(self.dimension)))
        return np.diag(fisher_diag)

    def export_dict(self, variational_params):
        """
        Create a dict of the distribution based on the given parameters.

        Args:
            variational_params (np.array): Variational parameters

        Returns:
            export_dict (dictionnary): Dict containing distribution information

        """
        mean, cov = self.reconstruct_parameters(variational_params)
        sd = cov ** 0.5
        export_dict = {
            "type": "meanfield_Normal",
            "mean": mean,
            "covariance": np.diag(cov),
            "standard_deviation": sd,
            "variational_parameters": variational_params,
        }
        return export_dict


class FullRankNormalVariational(VariationalDistribution):
    """
    Fullrank multivariate normal distribution. Uses the parameterization (as in [1])
    :math:`parameters=[\\mu, \\lambda]` where :math:`\\mu` are the mean values and 
    :math:`\\lambda` is an array containing the nonzero entries of the lower Cholesky 
    decomposition of the covariance matrix :math:`L`: 
    :math:`\\lambda=[L_{00},L_{10},L_{11},L_{20},L_{21},L_{22}, ...]`. 
    This allows the parameters :math:`\\lambda` to be unconstrained. 
        
    References:
        [1]: Kucukelbir, Alp, et al. "Automatic differentiation variational inference."
             The Journal of Machine Learning Research 18.1 (2017): 430-474.
    
    Attributes:
        dimension (int): Dimension of the random variable
        num_params (int): Number of parameters used in the parameterization
    """

    def __init__(self, dimension):
        super(FullRankNormalVariational, self).__init__(dimension)
        self.num_params = (dimension * (dimension + 1)) // 2 + dimension

    def initialize_parameters_randomly(self):
        """
        Initialize the variational parameters by
        :math:`\\mu=Uniform(-0.1,0.1)`
        :math:`L=diag(Uniform(0.9,1.1))` where :math:`\\Sigma=LL^T`

        Args:
            None

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
        """
        Construct the variational parameters for a given mean vector and covariance matrix  

        Args:
            mean (np.array): Mean values of the distribution
            covariance (np.array): Covariance of the distribution

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
        """
        Reconstruct mean value, covariance and its Cholesky decomposition of the distribution 
        based on variational parameters.

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

    def draw(self, variational_params, num_draws=1):
        """
        Draw `num_draw` samples from the variational distribution given variational parameters. 

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
        """
        Logpdf of the variational distribution evaluted using the variational parameters at given 
        samples `x`.
        
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
        """
        Pdf of the variational distribution evaluted using the variational parameters at given 
        samples `x`. First computes the logpdf, which numerically more stable for exponential 
        distributions.
        
        Args: 
            variational_params (np.array): Variational parameters 
            x (np.array): Row-wise samples

        Returns:
            pdf (np.array): Rowvector of the pdfs

        """
        pdf = np.exp(self.logpdf(variational_params, x))
        return pdf

    def grad_params_logpdf(self, variational_params, x):
        """
        Computes the gradient of the logpdf w.r.t. to the variational parameters of the 
        distribution evaluated at samples x. Also known as the score function.

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

    def fisher_information_matrix(self, variational_params):
        """
        Compute the Fisher information matrix analytically.

        Args:
            variational_params (np.array): Variational parameters

        Returns: 
            FIM (np.array): Matrix (num parameters x num parameters)

        """
        _, cov, L = self.reconstruct_parameters(variational_params)
        mu_block = np.linalg.inv(cov + 1e-8 * np.eye(len(cov)))

        n_params_chol = (self.dimension * (self.dimension + 1)) // 2
        sigma_block = np.zeros((n_params_chol, n_params_chol))
        A_list = []
        # Improvements of this implementation are welcomed!
        for r in range(0, self.dimension):
            for s in range(0, r + 1):
                A_q = np.zeros(L.shape)
                A_q[r, s] = 1
                A_q = np.matmul(L, A_q.T)
                A_q = np.linalg.solve(cov, A_q + A_q.T)
                A_list.append(A_q)
        for p in range(n_params_chol):
            for q in range(p + 1):
                val = 0.5 * np.trace(np.matmul(A_list[p], A_list[q]))
                sigma_block[p, q] = val
                sigma_block[q, p] = val

        return scipy.linalg.block_diag(mu_block, sigma_block)

    def export_dict(self, variational_params):
        """
        Create a dict of the distribution based on the given parameters.

        Args:
            variational_params (np.array): Variational parameters

        Returns:
            export_dict (dictionnary): Dict containing distribution information

        """
        mean, cov, _ = self.reconstruct_parameters(variational_params)
        export_dict = {
            "type": "fullrank_Normal",
            "mean": mean,
            "covariance": cov,
            "variational_parameters": variational_params,
        }
        return export_dict


def create_variational_distribution(distribution_options):
    """ 
    Create variational distribution object fromdictionary

    Args:
        distribution_options (dict): Dictionary containing parameters
                                     defining the distribution

    Returns:
        distribution:     Variational distribution object

    """
    distribution_family = distribution_options.get('variational_family', None)
    approximation_type = distribution_options.get('variational_approximation_type', None)
    if distribution_family == "normal":
        dimension = distribution_options.get('dimension')
        if approximation_type == "mean_field":
            distribution_obj = MeanFieldNormalVariational(dimension)
        elif approximation_type == "fullrank":
            distribution_obj = FullRankNormalVariational(dimension)
        else:
            supported_types = {'mean_field', 'fullrank'}
            raise ValueError(
                "Requested variational approximation type not supported: {0}.\n"
                "Supported types:  {1}. "
                "".format(approximation_type, supported_types)
            )

    else:
        supported_types = {'normal'}
        raise ValueError(
            "Requested variational family type not supported: {0}.\n"
            "Supported types:  {1}. "
            "".format(distribution_family, supported_types)
        )
    return distribution_obj

