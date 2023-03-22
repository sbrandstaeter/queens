"""Elementary Effects iterator module.

Elementary Effects (also called Morris method) is a global sensitivity
analysis method, which can be used for parameter fixing (ranking).
"""

import logging

import matplotlib as mpl
import numpy as np
from SALib.analyze import morris as morris_analyzer
from SALib.sample import morris

import pqueens.visualization.sa_visualization as qvis
from pqueens.distributions.uniform import UniformDistribution
from pqueens.iterators.iterator import Iterator
from pqueens.utils.process_outputs import write_results

if not mpl.get_backend().lower() == 'agg':
    mpl.use('TkAgg')

_logger = logging.getLogger(__name__)


class ElementaryEffectsIterator(Iterator):
    """Iterator to compute Elementary Effects (Morris method).

    Attributes:
        num_trajectories (int): Number of trajectories to generate.
        local_optimization (bool):  Flag whether to use local optimization according to Ruano et
                                    al. (2012). Speeds up the process tremendously for larger number
                                    of trajectories and *num_levels*. If set to *False*, brute force
                                    method is used.
        num_optimal_trajectories (int): Number of optimal trajectories to sample (between 2 and N).
        num_levels (int): Number of grid levels.
        seed (int): Seed for random number generation.
        confidence_level (float): Size of confidence interval.
        num_bootstrap_samples (int): Number of bootstrap samples used to compute confidence
                                     intervals for sensitivity measures.
        result_description (dict): Dictionary with desired result description.
        samples (np.array): Samples at which the model is evaluated.
        output (np.array): Results at samples.
        salib_problem (dict): Dictionary with SALib problem description.
        si (dict): Dictionary with all sensitivity indices.
    """

    def __init__(
        self,
        model,
        num_trajectories,
        local_optimization,
        num_optimal_trajectories,
        number_of_levels,
        seed,
        confidence_level,
        num_bootstrap_samples,
        result_description,
        global_settings,
    ):
        """Initialize ElementaryEffectsIterator.

        Args:
            model (model): QUEENS model to evaluate
            num_trajectories (int): number of trajectories to generate
            local_optimization (bool): flag whether to use local optimization according to Ruano
                                       et al. (2012). Speeds up the process tremendously for
                                       larger number of trajectories and num_levels. If set to
                                       ``False`` brute force method is used.
            num_optimal_trajectories (int): number of optimal trajectories to sample (between 2
                                            and N)
            number_of_levels (int): number of grid levels
            seed (int): seed for random number generation
            confidence_level (float): size of confidence interval
            num_bootstrap_samples (int): number of bootstrap samples used to compute confidence
                                         intervals for sensitivity measures
            result_description (dict): dictionary with desired result description
            global_settings (dict): Global settings of the QUEENS simulations
        """
        super().__init__(model, global_settings)
        self.num_trajectories = num_trajectories
        self.local_optimization = local_optimization
        self.num_optimal_trajectories = num_optimal_trajectories
        self.num_levels = number_of_levels
        self.seed = seed
        self.confidence_level = confidence_level
        self.num_bootstrap_samples = num_bootstrap_samples
        self.result_description = result_description

        self.samples = None
        self.output = None
        self.salib_problem = {}
        self.si = {}
        if result_description.get('plotting_options'):
            qvis.from_config_create(result_description.get('plotting_options'))

    def pre_run(self):
        """Generate samples for subsequent analysis and update model."""
        bounds = []
        for parameter in self.parameters.dict.values():
            if not isinstance(parameter, UniformDistribution) or parameter.dimension != 1:
                raise ValueError("Parameters must be 1D uniformly distributed.")
            bounds.append([parameter.lower_bound.squeeze(), parameter.upper_bound.squeeze()])

        self.salib_problem = {
            'num_vars': self.parameters.num_parameters,
            'names': self.parameters.names,
            'bounds': bounds,
            'groups': None,
        }

        self.samples = morris.sample(
            self.salib_problem,
            self.num_trajectories,
            num_levels=self.num_levels,
            optimal_trajectories=self.num_optimal_trajectories,
            local_optimization=self.local_optimization,
            seed=self.seed,
        )

    def core_run(self):
        """Run Analysis on model."""
        self.output = self.model.evaluate(self.samples)

        self.si = morris_analyzer.analyze(
            self.salib_problem,
            self.samples,
            np.reshape(self.output['mean'], (-1)),
            num_resamples=self.num_bootstrap_samples,
            conf_level=self.confidence_level,
            print_to_console=False,
            num_levels=self.num_levels,
            seed=self.seed,
        )

    def post_run(self):
        """Analyze the results."""
        results = self.process_results()
        if self.result_description is not None:
            self.print_results(results)

            if self.result_description["write_results"] is True:
                write_results(
                    results,
                    self.global_settings['output_dir'],
                    self.global_settings['experiment_name'],
                )

            qvis.sa_visualization_instance.plot(results)

    def process_results(self):
        """Write all results to self contained dictionary."""
        results = {"parameter_names": self.parameters.names, "sensitivity_indices": self.si}
        return results

    def print_results(self, results):
        """Print results to log.

        Args:
            results: TODO_doc
        """
        _logger.info(
            "{0:<20} {1:>10} {2:>10} {3:>15} {4:>10}".format(
                "Parameter", "Mu_Star", "Mu", "Mu_Star_Conf", "Sigma"
            )
        )

        for j in range(self.parameters.num_parameters):
            _logger.info(
                "{0!s:20} {1!s:10} {2!s:10} {3!s:15} {4!s:10}".format(
                    results['sensitivity_indices']['names'][j],
                    results['sensitivity_indices']['mu_star'][j],
                    results['sensitivity_indices']['mu'][j],
                    results['sensitivity_indices']['mu_star_conf'][j],
                    results['sensitivity_indices']['sigma'][j],
                )
            )
