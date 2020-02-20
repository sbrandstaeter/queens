"""collection of utility functions and classes for Markov Chain Monte Carlo algorithms"""

import abc
import numpy as np
import scipy.linalg
import scipy.stats


class ProposalDistribution:
    """ Base class for continuous probability distributions. """

    def __init__(self, mean, covariance, dimension):
        self.mean = mean
        self.covariance = covariance
        self.dimension = dimension

    @abc.abstractmethod
    def cdf(self, x):
        """
        Evaluate the cummulative distribution function (cdf).
        """
        pass

    @abc.abstractmethod
    def draw(self, num_draws=1):
        """ Draw num_draws samples from distribution. """
        pass

    @abc.abstractmethod
    def logpdf(self, x):
        """
        Evaluate the natural logarithm of the pdf at sample.
        """
        pass

    @abc.abstractmethod
    def pdf(self, x):
        """
        Evaluate the probability density function (pdf) at sample.
        """
        pass

    @abc.abstractmethod
    def ppf(self, x):
        """
        Evaluate the ppf, i.e. the inverse of the cdf.
        """
        pass


class Uniform(ProposalDistribution):
    """
    Uniform distribution.

    On the interval [a, b)
    """

    def __init__(self, a, b):

        if b <= a:
            raise ValueError(f"Invalid interval: [a, b) = [{a}, {b}).\n Set a<=b!")

        self.a = a
        self.b = b
        self.width = b - a

        mean = (self.a + self.b) / 2.0
        cov = self.width ** 2 / 12.0

        dim = 1
        super(Uniform, self).__init__(mean=mean, covariance=cov, dimension=dim)

        self.pdf_const = 1.0 / self.width
        self.logpdf_const = np.log(self.pdf_const)

    def cdf(self, x):
        return scipy.stats.uniform.cdf(x, loc=self.a, scale=self.width)

    def draw(self, num_draws=1):
        """
        Return ndarray of shape num_draws with samples between [a, b).
        """

        return np.random.uniform(low=self.a, high=self.b, size=num_draws)

    def logpdf(self, x):

        if x >= self.a and x < self.b:
            return self.logpdf_const
        else:
            return -np.inf

    def pdf(self, x):

        if x >= self.a and x < self.b:
            return self.pdf_const
        else:
            return 0

    def ppf(self, q):
        return scipy.stats.uniform.ppf(q=q, loc=self.a, scale=self.width)


class LogNormal(ProposalDistribution):
    """
    Lognormal distribution.

    mu and sigma are the parameters (mean and standard deviation) of the
    underlying normal distribution.
    """

    def __init__(self, mu, sigma):

        # sanity checks:
        if sigma <= 0:
            raise ValueError(f"Sigma has to be positiv. (sigma > 0). You supplied {sigma}")

        self.mu = mu
        self.sigma = sigma

        self.distr = scipy.stats.lognorm(scale=np.exp(self.mu), s=self.sigma)

        mean = np.exp(self.mu + self.sigma ** 2 / 2.0)
        cov = (np.exp(self.sigma ** 2) - 1) * np.exp(2 * self.mu + self.sigma ** 2)

        dim = 1
        super(LogNormal, self).__init__(mean, cov, dim)

        self.K1 = 1.0 / (self.sigma * np.sqrt(2 * np.pi))
        self.log_K1 = np.log(self.K1)

        self.K2 = 1.0 / (2.0 * self.sigma ** 2)

    def cdf(self, x):
        return scipy.stats.lognorm.cdf(x, s=self.sigma, scale=np.exp(self.mu))

    def draw(self, num_draws=1):
        return np.random.lognormal(mean=self.mu, sigma=self.sigma, size=num_draws)

    def logpdf(self, x):
        return -self.K2 * (np.log(x) - self.mu) ** 2 + self.log_K1 - np.log(x)

    def pdf(self, x):
        return self.K1 * np.exp(-self.K2 * (np.log(x) - self.mu) ** 2) / x

    def ppf(self, q):
        return scipy.stats.lognorm.ppf(q, s=self.sigma, scale=np.exp(self.mu))


class NormalProposal(ProposalDistribution):
    """ Normal distribution. """

    def __init__(self, mean, covariance):

        # convert to ndarrays if possible and not already done
        mean = np.array(mean)
        covariance = np.array(covariance)

        if covariance.size == 1:
            dimension = 1
        elif covariance.ndim == 2:
            size_dim_1, size_dim_2 = covariance.shape
            # safety checks
            if size_dim_1 != size_dim_2:
                raise ValueError("Covariance matrix is not quadratic.")
            if not np.allclose(covariance.T, covariance):
                raise ValueError("Covariance matrix is not symmetric.")
            dimension = size_dim_1
        else:
            raise ValueError("Covariance matrix has wrong dimension.")

        mean = np.atleast_1d(mean)
        if mean.ndim != 1:
            raise ValueError("Mean vector has wrong dimension.")

        if mean.shape[0] != dimension:
            raise ValueError("Mean vector has wrong length.")

        super(NormalProposal, self).__init__(mean, covariance, dimension)

        self.low_chol = scipy.linalg.cholesky(covariance, lower=True)
        # precision matrix Q and determinant of cov matrix
        if self.dimension == 1:
            self.Q = 1.0 / self.covariance
            self.det_cov = self.covariance
            self.std = np.sqrt(self.covariance)
        else:
            self.Q = np.linalg.inv(self.covariance)
            self.det_cov = np.linalg.det(self.covariance)
            self.std = np.NaN
        # constant needed for pdf
        self.K1 = 1.0 / (np.sqrt((2.0 * np.pi) ** self.dimension * self.det_cov))
        # constant needed for pdf
        self.log_K1 = np.log(self.K1)

    def cdf(self, x):
        if self.dimension == 1:
            return scipy.stats.norm.cdf(x, loc=self.mean, scale=self.std)
        else:
            return scipy.stats.multivariate_normal.cdf(x, mean=self.mean, cov=self.covariance)

    def draw(self, num_draws=1):
        uncorrelated_vector = (
            np.random.randn(self.dimension, num_draws)
            .reshape((self.dimension, num_draws))
            .reshape((self.dimension, num_draws))
        )
        return (
            np.reshape(self.mean, (self.dimension, 1))
            + (np.dot(self.low_chol, uncorrelated_vector))
        ).T

    def logpdf(self, x):
        y = x - self.mean
        logpdf = self.log_K1 - 0.5 * np.dot(np.dot(y, self.Q), y)
        return logpdf

    def ppf(self, q):
        if self.dimension == 1:
            return scipy.stats.norm.ppf(q, loc=self.mean, scale=self.std)
        else:
            raise RuntimeError(
                "ppf for multivariate gaussians is not supported.\n"
                "It is not uniquely defined, since cdf is not uniquely defined! "
            )

    def pdf(self, x):
        y = x - self.mean
        pdf = self.K1 * np.exp(-0.5 * np.dot(np.dot(y, self.Q), y))
        return pdf


def create_proposal_distribution(distribution_options):
    """ Create proposal distribution object from parameter dictionary

    Args:
        distribution_options (dict): Dictionary containing parameters
                                     defining the distribution

    Returns:
        distribution:     Distribution object
    """
    distribution_type = distribution_options.get('distribution', None)
    if distribution_type is None:
        distribution = None
    else:
        # TODO: what if we have  a distribution with more than two shape parameters
        shape_parameter_1, shape_parameter_2 = distribution_options.get(
            'distribution_parameter', list()
        )
        # shape_parameter_1 = np.squeeze(np.asarray(distribution_options['proposal_mean']))
        # shape_parameter_2 = np.squeeze(np.asarray(distribution_options['proposal_covariance']))
        if distribution_type == 'normal':
            distribution = NormalProposal(mean=shape_parameter_1, covariance=shape_parameter_2)
        elif distribution_type == 'uniform':
            distribution = Uniform(a=shape_parameter_1, b=shape_parameter_2)
        elif distribution_type == 'lognormal':
            distribution = LogNormal(mu=shape_parameter_1, sigma=shape_parameter_2)
        else:
            supported_proposal_types = {'normal', 'lognormal', 'uniform'}
            raise ValueError(
                "Requested distribution type not supported: {0}.\n"
                "Supported types of proposal distribution:  {1}. "
                "".format(distribution_type, supported_proposal_types)
            )
    return distribution


def mh_select(log_acceptance_probability, current_sample, proposed_sample):
    """Do Metropolis Hastings selection"""

    isfinite = np.isfinite(log_acceptance_probability)
    accept = (
        np.log(np.random.uniform(size=log_acceptance_probability.shape))
        < log_acceptance_probability
    )

    bool_idx = isfinite * accept

    selected_samples = np.where(bool_idx, proposed_sample, current_sample)

    return selected_samples, bool_idx


def tune_scale_covariance(scale_covariance, accept_rate):
    """
    Tune the acceptance rate according to the last tuning interval

    The goal is an acceptance rate within 20\% - 50\%.
    The (acceptance) rate is adapted according to the following rule:

        Acceptance Rate    Variance adaptation factor
        ---------------    --------------------------
        <0.001                       x 0.1
        <0.05                        x 0.5
        <0.2                         x 0.9
        >0.5                         x 1.1
        >0.75                        x 2
        >0.95                        x 10

    The implementation is modified from [1].

    Reference:
    [1]: https://github.com/pymc-devs/pymc3/blob/master/pymc3/step_methods/metropolis.py
    """

    scale_covariance = np.where(accept_rate < 0.001, scale_covariance * 0.1, scale_covariance)
    scale_covariance = np.where(
        (accept_rate >= 0.001) * (accept_rate < 0.05), scale_covariance * 0.5, scale_covariance
    )
    scale_covariance = np.where(
        (accept_rate >= 0.05) * (accept_rate < 0.2), scale_covariance * 0.9, scale_covariance
    )
    scale_covariance = np.where(
        (accept_rate > 0.5) * (accept_rate <= 0.75), scale_covariance * 1.1, scale_covariance
    )
    scale_covariance = np.where(
        (accept_rate > 0.75) * (accept_rate <= 0.95), scale_covariance * 2.0, scale_covariance
    )
    scale_covariance = np.where((accept_rate > 0.95), scale_covariance * 10.0, scale_covariance)

    return scale_covariance
