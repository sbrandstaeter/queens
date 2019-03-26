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
        return np.squeeze(np.dot(self.low_chol, uncorrelated_vector))


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

    if np.isfinite(log_acceptance_probability) and np.log(np.random.uniform()) < log_acceptance_probability:
        return proposed_sample, True

    return current_sample, False

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

    The implementation is taken from [1].

    Reference:
    [1]: https://github.com/pymc-devs/pymc3/blob/master/pymc3/step_methods/metropolis.py
    """

    # Switch statement
    if accept_rate < 0.001:
        # reduce by 90 percent
        scale_covariance *= 0.1
    elif accept_rate < 0.05:
        # reduce by 50 percent
        scale_covariance *= 0.5
    elif accept_rate < 0.2:
        # reduce by ten percent
        scale_covariance *= 0.9
    elif accept_rate > 0.95:
        # increase by factor of ten
        scale_covariance *= 10.0
    elif accept_rate > 0.75:
        # increase by double
        scale_covariance *= 2.0
    elif accept_rate > 0.5:
        # increase by ten percent
        scale_covariance *= 1.1

    return scale_covariance