"""Estimate Sobol indices."""
import logging
import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from SALib.analyze import sobol
from SALib.sample import saltelli

from pqueens.iterators.iterator import Iterator
from pqueens.models.model import Model
from pqueens.utils.process_outputs import write_results

_logger = logging.getLogger(__name__)


# TODO deal with non-uniform input distribution
class SobolIndexIterator(Iterator):
    """Sobol Index Iterator.

    This class essentially provides a wrapper around the SALib library

    Attributes:
        seed (int):                         Seed for random number generator
        num_samples (int):                  Number of samples
        calc_second_order (bool):           Calculate second-order sensitivities
        num_bootstrap_samples (int):        Number of bootstrap samples
        confidence_level (float):           The confidence interval level
        samples (np.array):                 Array with all samples
        output (dict)                       Dict with all outputs corresponding to
                                            samples
        salib_problem (dict):               Problem definition for SALib
        num_params (int):                   Number of parameters
        parameter_names (list):             List with parameter names
        sensitivity_indices (dict):         Dictionary with sensitivity indices
    """

    def __init__(
        self,
        model,
        seed,
        num_samples,
        calc_second_order,
        num_bootstrap_samples,
        confidence_level,
        result_description,
        global_settings,
    ):
        """Initialize Saltelli SALib iterator object.

        Args:
            model (model): Model to be evaluated by iterator
            seed (int): Seed for random number generation
            num_samples (int): Number of desired (random) samples
            calc_second_order (bool): Calculate second-order sensitivities
            num_bootstrap_samples (int): Number of bootstrap samples
            confidence_level (float): The confidence interval level
            result_description (dict): Dictionary with desired result description
            global_settings (dict): Dictionary with global settings for the analysis
        """
        super(SobolIndexIterator, self).__init__(model, global_settings)

        self.seed = seed
        self.num_samples = num_samples
        self.calc_second_order = calc_second_order
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level = confidence_level
        self.result_description = result_description

        self.samples = None
        self.output = None
        self.salib_problem = None
        self.num_params = None
        self.parameter_names = []
        self.sensitivity_indices = None

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """Create Saltelli SALib iterator from problem description.

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                     in options dict (optional)
            model (model): Model to iterate (optional)

        Returns:
            iterator: Saltelli SALib iterator object
        """
        if iterator_name is None:
            method_options = config["method"]["method_options"]
        else:
            method_options = config[iterator_name]["method_options"]

        if model is None:
            model_name = method_options["model"]
            model = Model.from_config_create_model(model_name, config)

        return cls(
            model,
            method_options["seed"],
            method_options["num_samples"],
            method_options["calc_second_order"],
            method_options["num_bootstrap_samples"],
            method_options["confidence_level"],
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
        dists = []
        self.num_params = 0
        for key, value in parameter_info["random_variables"].items():
            self.parameter_names.append(key)
            max_temp = value["distribution_parameter"][1]
            min_temp = value["distribution_parameter"][0]

            # in queens normal distributions are parameterized with mean and var
            # in salib normal distributions are parameterized via mean and std
            # -> we need to reparameterize normal distributions
            if value['distribution'] in ['normal']:
                max_temp = np.sqrt(max_temp)

            bounds.append([min_temp, max_temp])
            dist = self._get_sa_lib_distribution_name(value["distribution"])
            dists.append(dist)
            self.num_params += 1

        random_fields = parameter_info.get("random_fields", None)
        if random_fields is not None:
            raise RuntimeError(
                "The SaltelliIterator does not work in conjunction with random fields."
            )

        self.salib_problem = {
            'num_vars': self.num_params,
            'names': self.parameter_names,
            'bounds': bounds,
            'dists': dists,
        }

        _logger.info(f"Draw {self.num_samples} samples...")
        self.samples = saltelli.sample(
            self.salib_problem,
            self.num_samples,
            calc_second_order=self.calc_second_order,
            skip_values=1024,
        )
        _logger.debug(self.samples)

    def get_all_samples(self):
        """Return all samples."""
        return self.samples

    def core_run(self):
        """Run Analysis on model."""
        _logger.info("Evaluate model...")
        self.model.update_model_from_sample_batch(self.samples)
        self.output = self.eval_model()

        _logger.info("Calculate Sensitivity Indices...")
        self.sensitivity_indices = sobol.analyze(
            self.salib_problem,
            np.reshape(self.output['mean'], (-1)),
            calc_second_order=self.calc_second_order,
            num_resamples=self.num_bootstrap_samples,
            conf_level=self.confidence_level,
            print_to_console=False,
            seed=self.seed,
        )

    def post_run(self):
        """Analyze the results."""
        results = self.process_results()
        if self.result_description is not None:
            if self.result_description["write_results"] is True:
                write_results(
                    results,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )
            self.print_results(results)
            if self.result_description["plot_results"] is True:
                self.plot_results(results)

    def print_results(self, results):
        """Function to print results."""
        S = results["sensitivity_indices"]
        parameter_names = results["parameter_names"]

        additivity = np.sum(S["S1"])
        higher_interactions = 1 - additivity
        S_df = pd.DataFrame(
            {key: value for (key, value) in S.items() if key not in ["S2", "S2_conf"]},
            index=parameter_names,
        )
        _logger.info("Main and Total Effects:")
        _logger.info(S_df)
        _logger.info("Additivity, sum of main effects (for independent variables):")
        _logger.info(f'S_i = {additivity}')

        if self.calc_second_order:
            S2 = S["S2"]
            S2_conf = S["S2_conf"]

            for j in range(self.num_params):
                S2[j, j] = S["S1"][j]
                for k in range(j + 1, self.num_params):
                    S2[k, j] = S2[j, k]
                    S2_conf[k, j] = S2_conf[j, k]

            S2_df = pd.DataFrame(S2, columns=parameter_names, index=parameter_names)
            S2_conf_df = pd.DataFrame(S2_conf, columns=parameter_names, index=parameter_names)

            _logger.info("Second Order Indices (diagonal entries are main effects):")
            _logger.info(S2_df)
            _logger.info("Confidence Second Order Indices:")
            _logger.info(S2_conf_df)

            # we extract the upper triangular matrix which includes the diagonal entries
            # therefore we have to subtract the trace
            second_order_interactions = np.sum(np.triu(S2, k=1))
            higher_interactions = higher_interactions - second_order_interactions
            str_second_order_interactions = f'S_ij = {second_order_interactions}'

            _logger.info("Sum of second order interactions:")
            _logger.info(str_second_order_interactions)

            str_higher_order_interactions = f'1 - S_i - S_ij = {higher_interactions}'
        else:
            str_higher_order_interactions = f'1 - S_i = {higher_interactions}'

        _logger.info("Higher order interactions:")
        _logger.info(str_higher_order_interactions)

    @staticmethod
    def _get_sa_lib_distribution_name(distribution_name):
        """Convert QUEENS distribution name to SALib distribution name.

        Args:
            distribution_name (string): Name of distribution

        Returns:
            string: Name of distribution in SALib
        """
        if distribution_name == 'uniform':
            sa_lib_distribution_name = 'unif'
        elif distribution_name == 'normal':
            sa_lib_distribution_name = 'norm'
        elif distribution_name == 'lognormal':
            sa_lib_distribution_name = 'lognorm'
        else:
            valid_dists = ['uniform', 'normal', 'lognormal']
            raise ValueError('Distributions: choose one of %s' % ", ".join(valid_dists))
        return sa_lib_distribution_name

    def process_results(self):
        """Write all results to self contained dictionary."""
        results = {
            "parameter_names": self.parameter_names,
            "sensitivity_indices": self.sensitivity_indices,
            "second_order": self.calc_second_order,
            "samples": self.samples,
            "output": self.output,
        }

        return results

    def plot_results(self, results):
        """Create bar graph of first order sensitivity indices.

        Args:
            results   (dict):    Dictionary with results
        """
        experiment_name = self.global_settings["experiment_name"]

        # Plot first-order indices also called main effect
        chart_name = experiment_name + '_S1.html'
        chart_path = os.path.join(self.global_settings["output_dir"], chart_name)
        bars = go.Bar(
            x=results["parameter_names"],
            y=results["sensitivity_indices"]["S1"],
            error_y=dict(
                type='data', array=results["sensitivity_indices"]["S1_conf"], visible=True
            ),
        )
        data = [bars]

        layout = dict(
            title='First-Order Sensitivity Indices',
            xaxis=dict(title='Parameter'),
            yaxis=dict(title='Main Effect'),
        )

        fig = go.Figure(data=data, layout=layout)
        fig.write_html(chart_path)

        # Plot total indices also called total effect
        chart_name = experiment_name + '_ST.html'
        chart_path = os.path.join(self.global_settings["output_dir"], chart_name)
        bars = go.Bar(
            x=results["parameter_names"],
            y=results["sensitivity_indices"]["ST"],
            error_y=dict(
                type='data', array=results["sensitivity_indices"]["ST_conf"], visible=True
            ),
        )
        data = [bars]

        layout = dict(
            title='Total Sensitivity Indices',
            xaxis=dict(title='Parameter'),
            yaxis=dict(title='Total Effect'),
        )

        fig = go.Figure(data=data, layout=layout)
        fig.write_html(chart_path)

        # Plot second order indices (if applicable)
        if self.calc_second_order:
            S2 = results["sensitivity_indices"]["S2"]
            S2 = S2[np.triu_indices(self.num_params, k=1)]

            S2_conf = results["sensitivity_indices"]["S2_conf"]
            S2_conf = S2_conf[np.triu_indices(self.num_params, k=1)]

            # build list of names of second-order indices
            names = []
            for i in range(1, self.num_params + 1):
                for j in range(i + 1, self.num_params + 1):
                    names.append(f"S{i}{j}")

            chart_name = experiment_name + '_S2.html'
            chart_path = os.path.join(self.global_settings["output_dir"], chart_name)
            bars = go.Bar(x=names, y=S2, error_y=dict(type='data', array=S2_conf, visible=True))
            data = [bars]

            layout = dict(
                title='Second Order Sensitivity Indices',
                xaxis=dict(title='Parameter'),
                yaxis=dict(title='Second Order Effects'),
            )

            fig = go.Figure(data=data, layout=layout)
            fig.write_html(chart_path)
