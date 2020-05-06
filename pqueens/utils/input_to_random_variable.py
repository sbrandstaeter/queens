"""
Convenience module to get samples from a probability distribution.
"""

from pqueens.utils.mcmc_utils import create_proposal_distribution

# TODO write tests


def get_random_samples(description, num_samples):
    """
    Get random samples based on QUEENS description of distribution

    Args:
        description (dict):         Dictionary containing QUEENS distribution
                                    description
        num_samples (int):          Number of samples to generate

    Returns:
        np.array:                   Array with samples
    """

    distribution = create_proposal_distribution(description)
    samples = distribution.draw(num_draws=num_samples)

    return samples
