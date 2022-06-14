"""Polynomial chaos iterator.

Is a wrapper on the chaospy library.
"""
import logging

import chaospy as cp
import numpy as np
from chaospy.distributions.sampler.generator import (
    SAMPLER_NAMES as collocation_valid_sampling_rules,
)
from chaospy.quadrature.frontend import SHORT_NAME_TABLE as projection_node_location_rules

from pqueens.iterators.iterator import Iterator
from pqueens.models import from_config_create_model
from pqueens.utils.process_outputs import write_results
from pqueens.utils.valid_options_utils import get_option

_logger = logging.getLogger(__name__)


class PolynomialChaosIterator(Iterator):
    """Collocation-based polynomial chaos iterator.

    Attributes:
        seed  (int): Seed for random number generation
        num_collocation_points (int): Number of samples to compute
        sampling_rule (dict): Rule according to which samples are drawn
        polynomial_order (int): Order of polynomial expansion
        sparse (bool): For pseudo project, if true uses sparse collocation points
        polynomial_chaos_approach (str): Approach for the polynomial chaos approach
        distribution (cp.distribution): Joint input distribution
        result_description (dict): Description of desired results
    """

    def __init__(
        self,
        model,
        seed,
        num_collocation_points,
        sampling_rule,
        polynomial_order,
        sparse,
        polynomial_chaos_approach,
        distribution,
        result_description,
        global_settings,
    ):
        """Initialise polynomial chaos iterator.

        Args:
        model (model): Model to be evaluated by iterator
        seed  (int): Seed for random number generation
        num_collocation_points (int): Number of samples to compute
        sampling_rule (dict): Rule according to which samples are drawn
        polynomial_order (int): Order of polynomial expansion
        sparse (bool): For pseudo project, if true uses sparse collocation points
        polynomial_chaos_approach (str): Approach for the polynomial chaos approach
        distribution (cp.distribution): Joint input distribution
        result_description (dict): Description of desired results
        global_settings (dict, optional): Settings for the QUEENS run.
        """
        super().__init__(model, global_settings)
        self.seed = seed
        self.num_collocation_points = num_collocation_points
        self.sampling_rule = sampling_rule
        self.polynomial_order = polynomial_order
        self.result_description = result_description
        self.sparse = sparse
        self.polynomial_chaos_approach = polynomial_chaos_approach
        self.distribution = distribution
        self.samples = None
        self.expansions = None
        self.result_dict = None

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create PCM iterator from problem description.

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: PolynomialChaosIterator object
        """
        method_options = config[iterator_name]['method_options']
        if model is None:
            model_name = method_options['model']
            model = from_config_create_model(model_name, config)

        result_description = method_options.get('result_description', None)
        global_settings = config.get('global_settings', None)

        if config.get('external_geometry', None) is not None:
            raise NotImplementedError("External geometry not supported with this iterator!")

        seed = method_options.get("seed", 42)
        polynomial_chaos_approach = method_options.get("approach", None)
        valid_approaches = ["pseudo_spectral", "collocation"]
        if polynomial_chaos_approach not in valid_approaches or polynomial_chaos_approach is None:
            raise ValueError(
                f"Approach '{polynomial_chaos_approach}' unknown. Valid options are "
                f"{', '.join(valid_approaches)}."
            )

        num_collocation_points = method_options.get("num_collocation_points", None)
        if not isinstance(num_collocation_points, int) or num_collocation_points < 1:
            raise ValueError("Number of samples for the polynomial must be a positive integer!")

        sparse = method_options.get("sparse", None)

        sampling_rule = method_options.get("sampling_rule", None)
        if polynomial_chaos_approach == "collocation":
            valid_sampling_rules = collocation_valid_sampling_rules
        elif polynomial_chaos_approach == "pseudo_spectral":
            _logger.info(
                f"Maximum number of collocation points was set to {num_collocation_points}."
            )
            valid_sampling_rules = projection_node_location_rules
            if sampling_rule is None:
                # Chaospy default
                sampling_rule = "clenshaw_curtis"

            if not isinstance(sparse, bool):
                raise ValueError(
                    f"Sparse input attribute needs to be set to true or false, not to {sparse}"
                )

        sampling_rule = get_option(
            valid_sampling_rules,
            sampling_rule,
            "Chaospy sampling rule unknown",
        )
        polynomial_order = method_options.get("polynomial_order")
        if not isinstance(polynomial_order, int) or polynomial_order < 0:
            raise ValueError(
                f"Polynomial expansion order has to be a positive integer. You provided "
                f"{polynomial_order}"
            )

        parameters_dict = model.get_parameter()
        distribution = from_config_create_chaospy_joint_distribution(parameters_dict)
        return cls(
            model=model,
            seed=seed,
            num_collocation_points=num_collocation_points,
            sampling_rule=sampling_rule,
            polynomial_order=polynomial_order,
            result_description=result_description,
            global_settings=global_settings,
            sparse=sparse,
            polynomial_chaos_approach=polynomial_chaos_approach,
            distribution=distribution,
        )

    def initialize_run(self):
        """Initiliaze run."""
        np.random.seed(self.seed)

    def eval_model(self):
        """Evaluate the model."""
        return self.model.evaluate()['mean']

    def evaluate_forward_model(self, samples):
        """Evaluate forward model from samples.

        Args:
            samples (np.ndarray): Row-wise samples to evaluated the model

        Returns:
            (np.ndarray): model evaluations
        """
        self.model.update_model_from_sample_batch(samples)
        return self.eval_model()

    def core_run(self):
        """Core run for the polynomial chaos iterator."""
        _logger.info(f"Polynomial chaos using a {self.polynomial_chaos_approach} approach")
        if self.polynomial_chaos_approach == "collocation":
            polynomial_expansion, collocation_points = self._regression_based_pc()
        elif self.polynomial_chaos_approach == "pseudo_spectral":
            polynomial_expansion, collocation_points = self._projection_based_pc()

        mean = cp.E(polynomial_expansion, self.distribution)
        covariance = cp.Cov(polynomial_expansion, self.distribution)
        self.result_dict = {}
        self.result_dict['polynomial_expansion'] = polynomial_expansion
        self.result_dict['distribution'] = self.distribution
        self.result_dict['mean'] = mean
        self.result_dict['covariance'] = covariance
        self.result_dict['collocation_points'] = collocation_points
        self.result_dict['polynomial_chaos_approach'] = self.polynomial_chaos_approach

    def _projection_based_pc(self):
        """Projection based polynomial chaos.

        Computes the coefficients of the polynomial expansion using a projection approach.

        Returns:
            polynomial_expansion (cp.polynomial): Projected expansion
            nodes (np.array): Column-wise locations to evaluate the model
        """
        # Generate nodes and weights for the polynomial chaos method
        nodes, weights = cp.generate_quadrature(
            self.polynomial_order, self.distribution, sparse=self.sparse, rule=self.sampling_rule
        )
        num_collocation_points = len(nodes)
        if num_collocation_points > self.num_collocation_points:
            raise ValueError(
                f"Maximum number of collocation points were set to {self.num_collocation_points}."
                f"However the current setup with polynomial degree {self.polynomial_order} would"
                f" lead to {num_collocation_points} collocation points"
            )
        _logger.info(f"Number of collocation points: {num_collocation_points}")
        evaluations = self.evaluate_forward_model(nodes.T)

        # Generate the polynomial chaos expansion based on the distribution
        expansion = cp.generate_expansion(self.polynomial_order, self.distribution)

        polynomial_expansion = cp.fit_quadrature(expansion, nodes, weights, evaluations)

        return polynomial_expansion, nodes

    def _regression_based_pc(self):
        """Regression based polynomial chaos.

        Computes the coefficients of the polynomial expansion using regression.

        Returns:
            polynomial_expansion (cp.polynomial): Fitted polynomial expansion
            nodes (np.array): Column-wise locations to evaluate the model
        """
        # Generate samples for the polynomial regression
        collocation_points = self.distribution.sample(
            self.num_collocation_points, rule=self.sampling_rule
        )

        evaluations = self.evaluate_forward_model(collocation_points.T)

        # Generate the polynomial chaos expansion based on the distribution
        expansion = cp.generate_expansion(self.polynomial_order, self.distribution)

        polynomial_expansion = cp.fit_regression(expansion, collocation_points, evaluations)

        return polynomial_expansion, collocation_points

    def post_run(self):
        """Analyze the results."""
        if self.result_description is not None:
            if self.result_description["write_results"] is True:
                write_results(
                    self.result_dict,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )


def from_config_create_chaospy_joint_distribution(parameters_dict):
    """Get random variables in chaospy distribution format.

    Args:
        parameters_dict (dict): Dict with the random variables

    Returns:
        chaospy distribution
    """
    # get random Variables
    random_variables = parameters_dict.get('random_variables', None)

    variables = []

    # loop over rvs and create joint
    for _, rv in random_variables.items():
        rv_size = rv['size']
        if rv_size != 1:
            raise ValueError("Multidimensional random variables are not supported yet.")

        variable = from_config_create_chaospy_distribution(rv)

        variables.append(variable)

    # Pass the variables list as arguments
    return cp.J(*variables)


def from_config_create_chaospy_distribution(distribution_options):
    """Create chaospy distribution object from parameter dictionary.

    Args:
        distribution_options (dict): Dictionary containing parameters
                                 defining the distribution

    Returns:
        distribution:     Distribution object in chaospy format
    """
    distribution_type = distribution_options.get('distribution', None)
    if distribution_type is None:
        distribution = None
    else:
        if distribution_type == 'normal':
            distribution = cp.Normal(
                mu=distribution_options["mean"], sigma=distribution_options["covariance"]
            )
        elif distribution_type == 'uniform':
            distribution = cp.Uniform(
                lower=distribution_options["lower_bound"],
                upper=distribution_options["upper_bound"],
            )
        elif distribution_type == 'lognormal':
            distribution = cp.LogNormal(
                mu=distribution_options["normal_mean"],
                sigma=distribution_options["normal_covariance"],
            )
        elif distribution_type == 'beta':
            distribution = cp.Beta(
                alpha=distribution_options["a"],
                beta=distribution_options["b"],
                lower=distribution_options["lower_bound"],
                upper=distribution_options["upper_bound"],
            )
    return distribution
