"""Iterators.

The iterator package contains the implementation of several UQ and
optimization methods. Each of which is implemented in their own iterator
class. The iterator is therefor one of the central building blocks, as
the iterators orchestrate the evaluations on one or multiple models.
QUEENS also permits nesting of iterators to enable hierarchical methods
or surrogate based UQ approaches.
"""
from pqueens.iterators.baci_lm_iterator import BaciLMIterator
from pqueens.iterators.black_box_variational_bayes import BBVIIterator
from pqueens.iterators.bmfia_iterator import BMFIAIterator
from pqueens.iterators.bmfmc_iterator import BMFMCIterator
from pqueens.iterators.data_iterator import DataIterator
from pqueens.iterators.elementary_effects_iterator import ElementaryEffectsIterator
from pqueens.iterators.grid_iterator import GridIterator
from pqueens.iterators.lhs_iterator import LHSIterator
from pqueens.iterators.lhs_iterator_mf import MFLHSIterator
from pqueens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from pqueens.iterators.monte_carlo_iterator import MonteCarloIterator
from pqueens.iterators.optimization_iterator import OptimizationIterator
from pqueens.iterators.polynomial_chaos_iterator import PolynomialChaosIterator
from pqueens.iterators.reparameteriztion_based_variational_inference import RPVIIterator
from pqueens.iterators.sequential_monte_carlo_chopin import SequentialMonteCarloChopinIterator
from pqueens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from pqueens.iterators.single_sim_run_iterator import SingleSimRunIterator
from pqueens.iterators.sobol_index_gp_uncertainty_iterator import SobolIndexGPUncertaintyIterator
from pqueens.iterators.sobol_index_iterator import SobolIndexIterator
from pqueens.iterators.sobol_sequence_iterator import SobolSequenceIterator


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
    method_dict = {
        'lhs': LHSIterator,
        'lhs_mf': MFLHSIterator,
        'metropolis_hastings': MetropolisHastingsIterator,
        'monte_carlo': MonteCarloIterator,
        'optimization': OptimizationIterator,
        'read_data_from_file': DataIterator,
        'elementary_effects': ElementaryEffectsIterator,
        'polynomial_chaos': PolynomialChaosIterator,
        'sobol_indices': SobolIndexIterator,
        'sobol_indices_gp_uncertainty': SobolIndexGPUncertaintyIterator,
        'smc': SequentialMonteCarloIterator,
        'smc_chopin': SequentialMonteCarloChopinIterator,
        'sobol_sequence': SobolSequenceIterator,
        'sing_sim_run': SingleSimRunIterator,
        'bmfmc': BMFMCIterator,
        'grid': GridIterator,
        'baci_lm': BaciLMIterator,
        'bbvi': BBVIIterator,
        'bmfia': BMFIAIterator,
        'rpvi': RPVIIterator,
    }

    method_name = config[iterator_name]['method_name']
    iterator_class = method_dict[method_name]
    iterator = iterator_class.from_config_create_iterator(config, iterator_name, model)

    return iterator
