"""Distributions."""

VALID_TYPES = {
    'normal': ["queens.distributions.normal", "NormalDistribution"],
    'mean_field_normal': ["queens.distributions.mean_field_normal", "MeanFieldNormalDistribution"],
    'uniform': ["queens.distributions.uniform", "UniformDistribution"],
    'lognormal': ["queens.distributions.lognormal", "LogNormalDistribution"],
    'beta': ["queens.distributions.beta", "BetaDistribution"],
    'exponential': ['queens.distributions.exponential', 'ExponentialDistribution'],
    'free': ['queens.distributions.free', 'FreeVariable'],
    'mixture': ['queens.distributions.mixture', 'MixtureDistribution'],
    'categorical': ['queens.distributions.categorical', 'CategoricalDistribution'],
    'bernoulli': ['queens.distributions.bernoulli', 'BernoulliDistribution'],
    'multinomial': ['queens.distributions.multinomial', 'MultinomialDistribution'],
    'particles': ['queens.distributions.particles', 'ParticleDiscreteDistribution'],
    'uniform_discrete': ['queens.distributions.uniform_discrete', 'UniformDiscreteDistribution'],
}
