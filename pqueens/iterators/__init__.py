"""Iterators.

The iterator package contains the implementation of several UQ and
optimization methods. Each of which is implemented in their own iterator
class. The iterator is therefor one of the central building blocks, as
the iterators orchestrate the evaluations on one or multiple models.
QUEENS also permits nesting of iterators to enable hierarchical methods
or surrogate based UQ approaches.
"""
from pqueens.utils.import_utils import get_module_class


def from_config_create_iterator(config, iterator_name='method', model=None):
    """Create iterator from problem description.

    Args:
        config (dict):       Dictionary with QUEENS problem description
        iterator_name (str): Name of iterator to identify right section
                             in options dict (optional)
        model (model):       Model to use (optional)

    Returns:
        iterator: Iterator object
    """
    valid_types = {
        'lhs': ['.lhs_iterator', 'LHSIterator'],
        'lhs_mf': ['.lhs_iterator_mf', 'MFLHSIterator'],
        'metropolis_hastings': ['.metropolis_hastings_iterator', 'MetropolisHastingsIterator'],
        'monte_carlo': ['.monte_carlo_iterator', 'MonteCarloIterator'],
        'optimization': ['.optimization_iterator', 'OptimizationIterator'],
        'read_data_from_file': ['.data_iterator', 'DataIterator'],
        'elementary_effects': ['.elementary_effects_iterator', 'ElementaryEffectsIterator'],
        'polynomial_chaos': ['.polynomial_chaos_iterator', 'PolynomialChaosIterator'],
        'sobol_indices': ['.sobol_index_iterator', 'SobolIndexIterator'],
        'sobol_indices_gp_uncertainty': [
            '.sobol_index_gp_uncertainty_iterator',
            'SobolIndexGPUncertaintyIterator',
        ],
        'smc': ['.sequential_monte_carlo_iterator', 'SequentialMonteCarloIterator'],
        'smc_chopin': ['.sequential_monte_carlo_chopin', 'SequentialMonteCarloChopinIterator'],
        'sobol_sequence': ['.sobol_sequence_iterator', 'SobolSequenceIterator'],
        'sing_sim_run': ['.single_sim_run_iterator', 'SingleSimRunIterator'],
        'bmfmc': ['.bmfmc_iterator', 'BMFMCIterator'],
        'grid': ['.grid_iterator', 'GridIterator'],
        'baci_lm': ['.baci_lm_iterator', 'BaciLMIterator'],
        'bbvi': ['.black_box_variational_bayes', 'BBVIIterator'],
        'bmfia': ['.bmfia_iterator', 'BMFIAIterator'],
        'rpvi': ['.reparameteriztion_based_variational_inference', 'RPVIIterator'],
    }

    iterator_options = config.get(iterator_name)
    iterator_type = iterator_options.get("method_name")
    iterator_class = get_module_class(iterator_options, valid_types, iterator_type)
    iterator = iterator_class.from_config_create_iterator(config, iterator_name, model)

    return iterator
