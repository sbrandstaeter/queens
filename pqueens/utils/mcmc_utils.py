"""collection of utility functions and classes for Markov Chain Monte Carlo algorithms"""

import abc
import numpy as np
import scipy.linalg


class ProposalDistribution:
    """Base class of proposal distributions for MCMC algorithms"""

    def __init__(self, covariance, dimension):
        self.covariance = covariance
        self.dimension = dimension

    @abc.abstractmethod
    def draw(self, num_draws=1):
        """ Draw num_draws samples from proposal distribution"""

        pass


class NormalProposal(ProposalDistribution):
    """Normal proposal distribution for MCMC algorithms"""

    def __init__(self, covariance):
        if covariance.size is 1:
            dimension = 1
        elif covariance.ndim is 2:
            size_dim_1, size_dim_2 = covariance.shape
            # safety checks
            if size_dim_1 != size_dim_2:
                raise ValueError("Covariance matrix is not quadratic.")
            if not (covariance.T == covariance).all():
                raise ValueError("Covariance matrix is not symmetric.")
            dimension = size_dim_1
        else:
            raise ValueError("Wrong dimension of covariance matrix.")

        super(NormalProposal, self).__init__(covariance, dimension)

        self.low_chol = scipy.linalg.cholesky(covariance, lower=True)

    def draw(self, num_draws=1):
        uncorrelated_vector = np.random.randn(self.dimension, num_draws)
        return np.squeeze(np.dot(self.low_chol, uncorrelated_vector)).T


def create_proposal_distribution(proposal_options):
    """ Create proposal distribution object from parameter dictionary

    Args:
        proposal_options (dict):    Dictionary containing
                                    parameters defining the
                                    proposal distribution

    Returns:
        Proposal:     Proposal object
    """
    proposal_type = proposal_options.get('type', None)
    proposal_covariance = np.squeeze(np.asarray(proposal_options['proposal_covariance']))

    if proposal_type == 'normal':
        proposal_distribution = NormalProposal(proposal_covariance)
    else:
        supported_proposal_types = {'normal'}
        raise ValueError("Requested distribution type not supported: {0}.\n"
                         "Supported types of proposal distribution:  {1}. "
                         "".format(proposal_type, supported_proposal_types))
    return proposal_distribution


def mh_select(log_acceptance_probability, current_sample, proposed_sample):
    """Do Metropolis Hastings selection"""

    isfinite = np.isfinite(log_acceptance_probability)
    accept = np.log(np.random.uniform(size=log_acceptance_probability.shape)) < log_acceptance_probability

    bool_idx = isfinite * accept

    selected_samples =  np.where(bool_idx, proposed_sample, current_sample)

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
    scale_covariance = np.where((accept_rate >= 0.001) * (accept_rate < 0.05), scale_covariance * 0.5, scale_covariance)
    scale_covariance = np.where((accept_rate >= 0.05) * (accept_rate < 0.2), scale_covariance * 0.9, scale_covariance)
    scale_covariance = np.where((accept_rate > 0.5) * (accept_rate <= 0.75), scale_covariance * 1.1, scale_covariance)
    scale_covariance = np.where((accept_rate > 0.75) * (accept_rate <= 0.95), scale_covariance * 2.0, scale_covariance)
    scale_covariance = np.where((accept_rate > 0.95), scale_covariance * 10.0, scale_covariance)

    return scale_covariance