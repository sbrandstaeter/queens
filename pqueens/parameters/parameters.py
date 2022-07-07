"""Parameters module."""

import logging
import sys

import numpy as np

from pqueens.parameters.fields import RandomField

from .fields import from_config_create_random_field
from .variables import from_config_create_random_variable

_logger = logging.getLogger(__name__)

this = sys.modules[__name__]
this.parameters = None


def from_config_create_parameters(config, pre_processor=None):
    """Create a QUEENS parameter object from config.

    This construct follows the spirit of singleton design patterns
    Informally: there only exists one parameters instance

    Args:
        config (dict): Problem configuration
        pre_processor (obj, optional): pre-processor object to read coordinates of random field
                                       discretization
    """
    parameters_options = config.get('parameters', None)

    if parameters_options is not None:
        rv_dict = parameters_options.get('random_variables', {})
        rf_dict = parameters_options.get('random_fields', {})
        parameters_dict = {}
        parameters_keys = []
        num_parameters = 0

        for rv_name, rv_dict in rv_dict.items():
            parameters_dict[rv_name] = from_config_create_random_variable(rv_dict)
            parameters_keys = _add_parameters_keys(parameters_keys, rv_name, rv_dict['dimension'])
            num_parameters += rv_dict['dimension']

        random_field_flag = False
        for rf_name, rf_dict in rf_dict.items():
            parameters_dict[rf_name] = from_config_create_random_field(
                rf_dict, pre_processor.coords_dict[rf_name]
            )
            parameters_keys += parameters_dict[rf_name].coords['keys']
            random_field_flag = True
            num_parameters += parameters_dict[rf_name].dim_truncated

        this.parameters = Parameters(
            parameters_dict, parameters_keys, num_parameters, random_field_flag
        )


def _add_parameters_keys(parameters_keys, parameter_name, dimension):
    """Add parameter keys to existing parameter keys.

    If the dimension of a parameter is larger than one, a separate unique key is added for each parameter
    member.
    Example: If parameter x1 is 3-dimensional the keys x1_0, x1_1, x1_2 is added.
             If parameter x1 is 1-dimensional the key x1 is added.

    Args:
        parameters_keys (list): List of existing parameter keys
        parameter_name (str): Parameter name to be added
        dimension (int): dimension of Parameter

    Returns:
        parameters_keys (list): List of keys for all parameter members
    """
    if dimension == 1:
        parameters_keys.append(parameter_name)
    else:
        parameters_keys.extend([f"{parameter_name}_{i}" for i in range(dimension)])
    return parameters_keys


class Parameters:
    """Parameters class.

    Attributes:
        dict (dict): Random variables and random fields stored in a dict
        parameters_keys (list): List of keys for all parameter members
        num_parameters (int): Number of (truncated) parameters
        random_field_flag (bool): Specifies if random fields are used
        names (list): Parameter names
    """

    def __init__(self, parameters_dict, parameters_keys, num_parameters, random_field_flag):
        """Initialize parameters object.

        Args:
            parameters_dict (dict): Random variables and random fields stored in a dict
            parameters_keys (list): List of keys for all parameter members
            num_parameters (int): Number of (truncated) parameters
            random_field_flag (bool): Specifies if random fields are used
        """
        self.dict = parameters_dict
        self.parameters_keys = parameters_keys
        self.num_parameters = num_parameters
        self.random_field_flag = random_field_flag
        self.names = list(parameters_dict.keys())

    def draw_samples(self, num_samples):
        """Draw samples from all parameters.

        Returns:
            samples (np.ndarray): Drawn samples
        """
        samples = np.zeros((num_samples, self.num_parameters))
        current_index = 0
        for parameter in self.dict.values():
            samples[
                :, current_index : current_index + parameter.dimension
            ] = parameter.draw_samples(num_samples)
            current_index += parameter.dimension
        return samples

    def joint_logpdf(self, samples):
        """Evaluate the logpdf summed over all parameters.

        Returns:
            logpdf (np.ndarray): logpdf summed over all parameters
        """
        samples = samples.reshape(-1, self.num_parameters)
        logpdf = 0
        i = 0
        for parameter in self.dict.values():
            logpdf += parameter.distribution.logpdf(samples[:, i : i + parameter.dimension])
            i += parameter.dimension
        return logpdf

    def inverse_cdf_transform(self, samples):
        """Transform samples to unit interval.

        Args:
            samples (np.ndarray): Samples that should be transformed.

        Returns:
            transformed_samples (np.ndarray): Transformed samples
        """
        samples = samples.reshape(-1, self.num_parameters)
        transformed_samples = np.zeros(samples.shape)
        for i, parameter in enumerate(self.to_list()):
            if parameter.dimension != 1:
                raise ValueError("Only 1D Random variables can be transformed!")
            transformed_samples[:, i] = parameter.distribution.ppf(samples[:, i])
        return transformed_samples

    def sample_as_dict(self, sample):
        """Return sample as a dict.

        Args:
            sample (np.ndarray): A single sample

        Returns:
            sample_dict (dict): Dictionary containing sample members and the corresponding parameter
                                keys
        """
        sample_dict = {}
        sample = sample.reshape(-1)
        if self.random_field_flag:
            sample = self.expand_random_field_realization(sample)
        for j, key in enumerate(self.parameters_keys):
            sample_dict[key] = sample[j]
        return sample_dict

    def expand_random_field_realization(self, truncated_sample):
        """Expand truncated representation of random fields.

        Args:
            truncated_sample (np.ndarray): Truncated representation of sample

        Returns:
            sample_expanded (np.ndarray) Expanded representation of sample
        """
        sample_expanded = np.zeros(len(self.parameters_keys))
        index_truncated = 0
        index_expanded = 0
        for parameter in self.to_list():
            if isinstance(parameter, RandomField):
                sample_expanded[
                    index_expanded : index_expanded + parameter.dimension
                ] = parameter.expanded_representation(
                    truncated_sample[index_truncated : index_truncated + parameter.dim_truncated]
                )
                index_expanded += parameter.dimension
                index_truncated += parameter.dim_truncated
            else:
                sample_expanded[
                    index_expanded : index_expanded + parameter.dimension
                ] = truncated_sample[index_truncated : index_truncated + parameter.dimension]
                index_expanded += parameter.dimension
                index_truncated += parameter.dimension
        return sample_expanded

    def to_list(self):
        """Return parameters as list.

        Returns:
            parameter_list (list): List of parameters
        """
        parameter_list = list(self.dict.values())
        return parameter_list
