# -*- coding: utf-8 -*-
"""Regression Approximations.

This package contains a set of regression approximations, i.e., regression
models which are the essential building block for surrogate models. For the
actual implementation of the regression models external third party libraries
are used, such as `GPy`_, `GPFlow`_, and `TensorFlow`_.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension


 .. _GPy:
     https://github.com/SheffieldML/GPy
 .. _GPFlow:
     https://github.com/GPflow/GPflow
 .. _TensorFlow:
     https://www.tensorflow.org/
"""


def from_config_create_regression_approximation(config, approx_name, Xtrain, Ytrain):
    """Create approximation from options dict.

    Args:
        config (dict): Dictionary with problem description
        approx_name (str): Name of the approximation model
        Xtrain (npq.array):     Training inputs
        Ytrain (np.array):     Training outputs

    Returns:
        regression_approximation (obj): Approximation object
    """
    from pqueens.utils.import_utils import get_module_attribute
    from pqueens.utils.valid_options_utils import get_option

    from .bayesian_neural_network import GaussianBayesianNeuralNetwork
    from .gp_approximation_gpflow import GPFlowRegression
    from .gp_approximation_gpflow_svgp import GPflowSVGP
    from .gp_approximation_gpy import GPGPyRegression
    from .gp_approximation_precompiled import GPPrecompiled
    from .heteroskedastic_GPflow import HeteroskedasticGP

    approx_dict = {
        'gp_approximation_gpy': GPGPyRegression,
        'heteroskedastic_gp': HeteroskedasticGP,
        'gp_approximation_gpflow': GPFlowRegression,
        'gaussian_bayesian_neural_network': GaussianBayesianNeuralNetwork,
        'gp_precompiled': GPPrecompiled,
        'gp_approximation_gpflow_svgp': GPflowSVGP,
    }
    approx_options = config[approx_name]
    if approx_options.get("external_python_module"):
        module_path = approx_options["external_python_module"]
        module_attribute = approx_options.get("type")
        approximation_class = get_module_attribute(module_path, module_attribute)
    else:
        approximation_class = get_option(approx_dict, approx_options.get("type"))

    return approximation_class.from_config_create(config, approx_name, Xtrain, Ytrain)
