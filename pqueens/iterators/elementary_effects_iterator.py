"""Elementary Effects iterator module.

Elementary Effects also called Morris method is a global sensitivity
analysis method which can be used for parameter fixing (ranking).
"""

import logging

import matplotlib as mpl
import numpy as np
from SALib.analyze import morris as morris_analyzer
from SALib.sample import morris

import pqueens.visualization.sa_visualization as qvis
from pqueens.iterators.iterator import Iterator
from pqueens.models import from_config_create_model
from pqueens.utils.process_outputs import write_results

if not mpl.get_backend().lower() == 'agg':
    mpl.use('TkAgg')

_logger = logging.getLogger(__name__)


class ElementaryEffectsIterator(Iterator):
    """Iterator to compute Elementary Effects (Morris method).

    Attributes:
        num_trajectories (int): number of trajectories to generate
        local_optimization (bool):  flag whether to use local optimization according to Ruano et
                                    al. (2012). Speeds up the process tremendously for larger number
                                    of trajectories and num_levels. If set to ``False`` brute force
                                    method is used.
        num_optimal_trajectories (int): number of optimal trajectories to sample (between 2 and N)
        num_levels (int): number of grid levels
        seed (int): seed for random number generation
        confidence_level (float): size of confidence interval
        num_bootstrap_samples (int): number of bootstrap samples used to compute confidence
                                     intervals for sensitivity measures
        num_params (int): number of model parameters
        parameter_names (list): list with parameter names
        samples (np.array): samples at which model is evaluated
        output (np.array): results at samples
        salib_problem (dict): dictionary with SALib problem description
        si (dict): Dictionary with all sensitivity indices
    """

    def __init__(
        self,
        model,
        num_trajectories,
        local_optimization,
        num_optimal_trajectories,
        num_levels,
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
            num_levels (int): number of grid levels
            seed (int): seed for random number generation
            confidence_level (float): size of confidence interval
            num_bootstrap_samples (int): number of bootstrap samples used to compute confidence
                                         intervals for sensitivity measures
            result_description (dict): dictionary with desired result description
            global_settings (dict): dictionary with global (all) settings of the analysis
        """
        super(ElementaryEffectsIterator, self).__init__(model, global_settings)
        self.num_trajectories = num_trajectories
        self.local_optimization = local_optimization
        self.num_optimal_trajectories = num_optimal_trajectories
        self.num_levels = num_levels
        self.seed = seed
        self.confidence_level = confidence_level
        self.num_bootstrap_samples = num_bootstrap_samples
        self.result_description = result_description

        self.num_params = None
        self.parameter_names = []
        self.samples = None
        self.output = None
        self.salib_problem = {}
        self.si = {}

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create ElementaryEffectsIterator iterator from problem description.

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model): Model to iterate (optional)

        Returns:
            iterator: ElementaryEffectsIterator iterator object
        """
        qvis.from_config_create(config)

        method_options = config[iterator_name]['method_options']

        if model is None:
            model_name = method_options["model"]
            model = from_config_create_model(model_name, config)

        if "num_traj_chosen" not in method_options:
            method_options["num_traj_chosen"] = None

        return cls(
            model,
            method_options["num_trajectories"],
            method_options["local_optimization"],
            method_options["num_optimal_trajectories"],
            method_options["number_of_levels"],
            method_options["seed"],
            method_options["confidence_level"],
            method_options["num_bootstrap_samples"],
            method_options.get("result_description", None),
            config["global_settings"],
        )

    def eval_model(self):
        """Evaluate the model."""
        return self.model.evaluate()

    def pre_run(self):
        """Generate samples for subsequent analysis and update model."""
        parameter_info = self.model.get_parameter()

        # setup SALib problem dict
        bounds = []
        self.num_params = 0
        for key, value in parameter_info["random_variables"].items():
            self.parameter_names.append(key)
            max_temp = value["upper_bound"]
            min_temp = value["lower_bound"]
            bounds.append([min_temp, max_temp])
            if "distribution" in value:
                raise ValueError("Parameters must not have probability distributions")
            self.num_params += 1

        if parameter_info.get("random_fields", None) is not None:
            raise RuntimeError(
                "Morris screening is currently not implemented in conjunction with random fields."
            )

        self.salib_problem = {
            'num_vars': self.num_params,
            'names': self.parameter_names,
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
        self.model.update_model_from_sample_batch(self.samples)

        self.output = self.eval_model()

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
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )

            qvis.sa_visualization_instance.plot(results)

    def process_results(self):
        """Write all results to self contained dictionary."""
        results = {"parameter_names": self.parameter_names, "sensitivity_indices": self.si}
        return results

    def print_results(self, results):
        """Print results to log."""
        _logger.info(
            "{0:<20} {1:>10} {2:>10} {3:>15} {4:>10}".format(
                "Parameter", "Mu_Star", "Mu", "Mu_Star_Conf", "Sigma"
            )
        )

        for j in range(self.num_params):
            _logger.info(
                "{0!s:20} {1!s:10} {2!s:10} {3!s:15} {4!s:10}".format(
                    results['sensitivity_indices']['names'][j],
                    results['sensitivity_indices']['mu_star'][j],
                    results['sensitivity_indices']['mu'][j],
                    results['sensitivity_indices']['mu_star_conf'][j],
                    results['sensitivity_indices']['sigma'][j],
                )
            )
