"""Likelihood models.

This package contains different likelihood models that can be used
QUEENS, to build probabilistic models. A standard use-case are inverse
problems.
"""

from pqueens.utils.import_utils import get_module_attribute
from pqueens.utils.valid_options_utils import get_option


def from_config_create_model(model_name, config):
    """Create a likelihood model from the problem description.

    Args:
        model_name (str): Name of the model
        config (dict): Dictionary with the problem description

    Returns:
        likelihood_model (obj): Instance of likelihood_model class
    """
    # get child likelihood classes
    from .bayesian_mf_gaussian_likelihood import BMFGaussianModel
    from .gaussian_likelihood import GaussianLikelihood

    model_dict = {
        'gaussian': GaussianLikelihood,
        'bmf_gaussian': BMFGaussianModel,
    }

    # get options
    model_options = config[model_name]
    if model_options.get("external_python_module"):
        module_path = model_options["external_python_module"]
        module_attribute = model_options.get("subtype")
        model_class = get_module_attribute(module_path, module_attribute)
    else:
        model_class = get_option(model_dict, model_options.get("subtype"))

    likelihood_model = model_class.from_config_create_likelihood(model_name, config)

    return likelihood_model
