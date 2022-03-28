"""Iterators.

The iterator package contains the implementation of several UQ and
optimization methods. Each of which is implemented in their own iterator
class. The iterator is therefor one of the central building blocks, as
the iterators orchestrate the evaluations on one or multiple models.
QUEENS also permits nesting of iterators to enable hierarchical methods
or surrogate based UQ approaches.
"""


def from_config_create_iterator(config, iterator_name=None, model=None):
    """Create iterator from problem description.

    Args:
        config (dict):       Dictionary with QUEENS problem description
        iterator_name (str): Name of iterator to identify right section
                             in options dict (optional)
        model (model):       Model to use (optional)

    Returns:
        iterator: Iterator object
    """
    from .baci_lm_iterator import BaciLMIterator
    from .black_box_variational_bayes import BBVIIterator
    from .bmfia_iterator import BMFIAIterator
    from .bmfmc_iterator import BMFMCIterator
    from .data_iterator import DataIterator
    from .elementary_effects_iterator import ElementaryEffectsIterator
    from .grid_iterator import GridIterator
    from .lhs_iterator import LHSIterator
    from .lhs_iterator_mf import MF_LHSIterator
    from .metropolis_hastings_iterator import MetropolisHastingsIterator
    from .monte_carlo_iterator import MonteCarloIterator
    from .optimization_iterator import OptimizationIterator
    from .sequential_monte_carlo_chopin import SequentialMonteCarloChopinIterator
    from .sequential_monte_carlo_iterator import SequentialMonteCarloIterator
    from .single_sim_run_iterator import SingleSimRunIterator
    from .sobol_index_gp_uncertainty_iterator import SobolIndexGPUncertaintyIterator
    from .sobol_index_iterator import SobolIndexIterator
    from .sobol_sequence_iterator import SobolSequenceIterator
    from .variational_inference_reparameterization import VIRPIterator

    method_dict = {
        'lhs': LHSIterator,
        'lhs_mf': MF_LHSIterator,
        'metropolis_hastings': MetropolisHastingsIterator,
        'monte_carlo': MonteCarloIterator,
        'optimization': OptimizationIterator,
        'read_data_from_file': DataIterator,
        'elementary_effects': ElementaryEffectsIterator,
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
        'virp': VIRPIterator,
    }

    if iterator_name is None:
        method_name = config['method']['method_name']
        iterator_class = method_dict[method_name]
        iterator = iterator_class.from_config_create_iterator(config, model)
    else:
        method_name = config[iterator_name]['method_name']
        iterator_class = method_dict[method_name]
        iterator = iterator_class.from_config_create_iterator(config, iterator_name, model)

    return iterator
