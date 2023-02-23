# -*- coding: utf-8 -*-
"""Regression Approximations.

This package contains a set of regression approximations, i.e. regression
models, which are the essential building block for surrogate models. For the
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
from pqueens.utils.import_utils import get_module_class

VALID_TYPES = {
    'gp_approximation_gpy': [
        'pqueens.regression_approximations.gp_approximation_gpy',
        'GPGPyRegression',
    ],
    'heteroskedastic_gp': [
        'pqueens.regression_approximations.heteroskedastic_GPflow',
        'HeteroskedasticGP',
    ],
    'gp_approximation_gpflow': [
        'pqueens.regression_approximations.gp_approximation_gpflow',
        'GPFlowRegression',
    ],
    'gaussian_bayesian_neural_network': [
        'pqueens.regression_approximations.bayesian_neural_network',
        'GaussianBayesianNeuralNetwork',
    ],
    'gp_precompiled': [
        'pqueens.regression_approximations.gp_approximation_precompiled',
        'GPPrecompiled',
    ],
    'gp_approximation_gpflow_svgp': [
        'pqueens.regression_approximations.gp_approximation_gpflow_svgp',
        'GPflowSVGP',
    ],
    'gaussian_nn': [
        'pqueens.regression_approximations.gaussian_neural_network',
        'GaussianNeuralNetwork',
    ],
}


def from_config_create_regression_approximation(config, approx_name, x_train, y_train):
    """Create approximation from options *dict*.

    Args:
        config (dict): Dictionary with problem description
        approx_name (str): Name of the approximation model
        x_train (np.array):     Training inputs
        y_train (np.array):     Training outputs

    Returns:
        regression_approximation (obj): Approximation object
    """
    approx_options = config[approx_name]
    approx_class = get_module_class(approx_options, VALID_TYPES)
    approx_obj = approx_class.from_config_create(config, approx_name, x_train, y_train)
    return approx_obj
