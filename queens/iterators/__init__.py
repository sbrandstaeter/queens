"""Iterators.

The iterator package contains the implementation of several UQ and
optimization methods, each of which is implemented in their own iterator
class. The iterator is therefor one of the central building blocks, as
the iterators orchestrate the evaluations on one or multiple models.
QUEENS also permits nesting of iterators to enable hierarchical methods
or surrogate based UQ approaches.
"""

VALID_TYPES = {
    'hmc': [
        'queens.iterators.hmc_iterator',
        'HMCIterator',
    ],
    'lhs': ['queens.iterators.lhs_iterator', 'LHSIterator'],
    'lhs_mf': ['queens.iterators.lhs_iterator_mf', 'MFLHSIterator'],
    'metropolis_hastings': [
        'queens.iterators.metropolis_hastings_iterator',
        'MetropolisHastingsIterator',
    ],
    'metropolis_hastings_pymc': [
        'queens.iterators.metropolis_hastings_pymc_iterator',
        'MetropolisHastingsPyMCIterator',
    ],
    'monte_carlo': ['queens.iterators.monte_carlo_iterator', 'MonteCarloIterator'],
    'nuts': [
        'queens.iterators.nuts_iterator',
        'NUTSIterator',
    ],
    'optimization': ['queens.iterators.optimization_iterator', 'OptimizationIterator'],
    'read_data_from_file': ['queens.iterators.data_iterator', 'DataIterator'],
    'elementary_effects': [
        'queens.iterators.elementary_effects_iterator',
        'ElementaryEffectsIterator',
    ],
    'polynomial_chaos': ['queens.iterators.polynomial_chaos_iterator', 'PolynomialChaosIterator'],
    'sobol_indices': ['queens.iterators.sobol_index_iterator', 'SobolIndexIterator'],
    'sobol_indices_gp_uncertainty': [
        'queens.iterators.sobol_index_gp_uncertainty_iterator',
        'SobolIndexGPUncertaintyIterator',
    ],
    'smc': ['queens.iterators.sequential_monte_carlo_iterator', 'SequentialMonteCarloIterator'],
    'smc_chopin': [
        'queens.iterators.sequential_monte_carlo_chopin',
        'SequentialMonteCarloChopinIterator',
    ],
    'sobol_sequence': ['queens.iterators.sobol_sequence_iterator', 'SobolSequenceIterator'],
    'points': ['queens.iterators.points_iterator', 'PointsIterator'],
    'bmfmc': ['queens.iterators.bmfmc_iterator', 'BMFMCIterator'],
    'grid': ['queens.iterators.grid_iterator', 'GridIterator'],
    'baci_lm': ['queens.iterators.baci_lm_iterator', 'BaciLMIterator'],
    'bbvi': ['queens.iterators.black_box_variational_bayes', 'BBVIIterator'],
    'bmfia': ['queens.iterators.bmfia_iterator', 'BMFIAIterator'],
    'rpvi': ['queens.iterators.reparameteriztion_based_variational_inference', 'RPVIIterator'],
    'classification': ['queens.iterators.classification', 'ClassificationIterator'],
}
