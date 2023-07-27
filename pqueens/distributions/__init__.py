"""Distributions."""

VALID_TYPES = {
    'normal': ["pqueens.distributions.normal", "NormalDistribution"],
    'mean_field_normal': ["pqueens.distributions.mean_field_normal", "MeanFieldNormalDistribution"],
    'uniform': ["pqueens.distributions.uniform", "UniformDistribution"],
    'lognormal': ["pqueens.distributions.lognormal", "LogNormalDistribution"],
    'beta': ["pqueens.distributions.beta", "BetaDistribution"],
    'exponential': ['pqueens.distributions.exponential', 'ExponentialDistribution'],
    'free': ['pqueens.distributions.free', 'FreeVariable'],
    'mixture': ['pqueens.distributions.mixture', 'MixtureDistribution'],
    'categorical': ['pqueens.distributions.categorical', 'CategoricalDistribution'],
    'bernoulli': ['pqueens.distributions.bernoulli', 'BernoulliDistribution'],
    'multinomial': ['pqueens.distributions.multinomial', 'MultinomialDistribution'],
    'particles': ['pqueens.distributions.particles', 'ParticleDiscreteDistribution'],
    'uniform_discrete': ['pqueens.distributions.uniform_discrete', 'UniformDiscreteDistribution'],
}
