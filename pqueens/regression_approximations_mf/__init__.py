# -*- coding: utf-8 -*-
"""Multi-Fidelity Regression Approximations.

This package contains a set of multi-fidelity regression approximations,
i.e. multi-task regression models which are the essential building block for
multi-fidelity surrogate models. For the actual implementation of the regression
models external third party libraries are used, such as `GPy`_, `GPFlow`_.

 .. _GPy:
     https://github.com/SheffieldML/GPy
 .. _GPFlow:
     https://github.com/GPflow/GPflow
"""
from pqueens.utils.import_utils import get_module_class

VALID_TYPES = {
    'mf_icm_gp_approximation_gpy': [
        'pqueens.regression_approximations_mf.mf_icm_gp_regression',
        'MF_ICM_GP_Regression',
    ],
    'mf_nar_gp_approximation_gpy_2_levels': [
        'pqueens.regression_approximations_mf.mf_nar_gp_regression_2_levels',
        'MF_NAR_GP_Regression_2_Levels',
    ],
    'mf_nar_gp_approximation_gpy_3_levels': [
        'pqueens.regression_approximations_mf.mf_nar_gp_regression_3_levels',
        'MF_NAR_GP_Regression_3_Levels',
    ],
}


def from_comfig_create_regression_approximators_mf(approx_options, x_train, y_train):
    """Create multi-fidelity approximation from options dict.

    Args:
        approx_options (dict): Dictionary with approximation options
        x_train (list):         List with training input arrays
        y_train (list):         List with training output arrays

    Returns:
        RegressionApproximationMF: Multi-fidelity regression approximation object
    """
    approx_class = get_module_class(approx_options, VALID_TYPES)
    approx_obj = approx_class.from_options(approx_options, x_train, y_train)
    return approx_obj
