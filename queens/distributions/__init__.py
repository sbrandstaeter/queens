"""Distributions."""

from queens.distributions.bernoulli import BernoulliDistribution
from queens.distributions.beta import BetaDistribution
from queens.distributions.categorical import CategoricalDistribution
from queens.distributions.exponential import ExponentialDistribution
from queens.distributions.free import FreeVariable
from queens.distributions.lognormal import LogNormalDistribution
from queens.distributions.mean_field_normal import MeanFieldNormalDistribution
from queens.distributions.multinomial import MultinomialDistribution
from queens.distributions.normal import NormalDistribution
from queens.distributions.particles import ParticleDiscreteDistribution
from queens.distributions.uniform import UniformDistribution
from queens.distributions.uniform_discrete import UniformDiscreteDistribution

VALID_TYPES = {
    'normal': NormalDistribution,
    'mean_field_normal': MeanFieldNormalDistribution,
    'uniform': UniformDistribution,
    'lognormal': LogNormalDistribution,
    'beta': BetaDistribution,
    'exponential': ExponentialDistribution,
    'free': FreeVariable,
    'categorical': CategoricalDistribution,
    'bernoulli': BernoulliDistribution,
    'multinomial': MultinomialDistribution,
    'particles': ParticleDiscreteDistribution,
    'uniform_discrete': UniformDiscreteDistribution,
}
