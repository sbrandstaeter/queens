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
        return np.dot(self.low_chol, uncorrelated_vector)


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
