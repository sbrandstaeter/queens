# -*- coding: utf-8 -*-
"""Multi-Fidelity Regression Approximations.

This package contains a set of multi-fidelits regression approximations,
i.e., multi-task regression models which are the essential building block for
multi-fidelity surrogate models. For the actual implementation of the regression
models external third party libraries are used, such as `GPy`_, `GPFlow`_.

 .. _GPy:
     https://github.com/SheffieldML/GPy
 .. _GPFlow:
     https://github.com/GPflow/GPflow
"""


def from_comfig_create_regression_approximators_mf(approx_options, Xtrain, Ytrain):
    """Create multi-fideltiy approximation from options dict.

    Args:
        approx_options (dict): Dictionary with approximation options
        Xtrain (list):         List with training input arrays
        Ytrain (list):         List with training output arrays

    Returns:
        RegressionApproximationMF: Multi-Fidelity regression approximation object
    """
    from .mf_icm_gp_regression import MF_ICM_GP_Regression
    from .mf_nar_gp_regression_2_levels import MF_NAR_GP_Regression_2_Levels
    from .mf_nar_gp_regression_3_levels import MF_NAR_GP_Regression_3_Levels

    approx_dict = {
        'mf_icm_gp_approximation_gpy': MF_ICM_GP_Regression,
        'mf_nar_gp_approximation_gpy_2_levels': MF_NAR_GP_Regression_2_Levels,
        'mf_nar_gp_approximation_gpy_3_levels': MF_NAR_GP_Regression_3_Levels,
    }

    approximation_class = approx_dict[approx_options["type"]]

    return approximation_class.from_options(approx_options, Xtrain, Ytrain)
