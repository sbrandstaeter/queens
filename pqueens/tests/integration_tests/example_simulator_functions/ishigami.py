"""Ishigami function is a three dimensional test function for sensitvity
analysis and UQ.

It is nonlinear and nonmonoton.
"""

import numpy as np

# default parameter values
P1 = 7
P2 = 0.1


def ishigami(x1, x2, x3, p1=P1, p2=P2):
    """Three dimensional benchmark function.

    Three dimensional benchmark function from [2] used for UQ because it
    exhibits strong nonlinearity and nonmonotonicity.
    It also has a peculiar dependence on x3, as described [5]).
    The values of a and b used in [1] and [3] are: p_1 = a = 7
    and p_2 = b = 0.1. In [5] the values a = 7 and b = 0.05 are used.

    The function is defined as:

     :math:`f({\\bf x}) = \\sin(x_1) + p_1 \\sin^2(x_2 +p_2x_3^4\\sin(x_1))`

    Typically distributions of the input random variables are is:
    :math:`x_i ~` Uniform[:math:`-\pi, \pi`], for all i = 1, 2, 3

    Args:
        x1 (float): Input parameter 1
        x2 (float): Input parameter 2
        x3 (float): Input parameter 3
        p1 (float): Coefficient (optional), with default value 7
        p2 (float): Coefficient (optional), with default value 0.1

    Returns:
        float : Value of ishigami function at [x1, x2, x3]

    References:

        [1] Crestaux, T., Martinez, J.-M., Le Maitre, O., & Lafitte, O. (2007).
            Polynomial chaos expansion for uncertainties quantification and
            sensitivity analysis [PowerPoint slides]. Retrieved from
            SAMO 2007 website: http://samo2007.chem.elte.hu/lectures/Crestaux.pdf.

        [2] Ishigami, T., & Homma, T. (1990, December). An importance
            quantification technique in uncertainty analysis for computer models.
            In Uncertainty Modeling and Analysis, 1990. Proceedings.,

        [3] Marrel, A., Iooss, B., Laurent, B., & Roustant, O. (2009).
            Calculations of sobol indices for the gaussian process metamodel.
            Reliability Engineering & System Safety, 94(3), 742-751.

        [4] Saltelli, A., Chan, K., & Scott, E. M. (Eds.). (2000).
            Sensitivity analysis (Vol. 134). New York: Wiley.

        [5] Sobol, I. M., & Levitan, Y. L. (1999). On the use of variance
            reducing multipliers in Monte Carlo computations of a global
            sensitivity index. Computer Physics Communications, 117(1), 52-61.
    """

    term1 = np.sin(x1)
    term2 = p1 * (np.sin(x2)) ** 2
    term3 = p2 * x3 ** 4 * np.sin(x1)

    return term1 + term2 + term3


def variance(p1=P1, p2=P2):
    """Variance of Ishigami test funcion.

    according to (50) in [1].

    [1] Homma, T., & Saltelli, A. (1996). Importance measures in global
    sensitivity analysis of nonlinear models. Reliability Engineering &
    System Safety, 52(1), 1–17.
    https://doi.org/10.1016/0951-8320(96)00002-6
    Args:
        p1 (float): Coefficient (optional), with default value 7
        p2 (float): Coefficient (optional), with default value 0.1

    Returns:
        float : Value of variance of ishigami function
    """
    return 0.125 * p1 ** 2 + 0.2 * p2 * np.pi ** 4 + p2 ** 2 * np.pi ** 8 / 18 + 0.5


def first_effect_variance(p1=P1, p2=P2):
    """Total variance of Ishigami test funcion.

    according to (50)-(53) in [1].

    [1] Homma, T., & Saltelli, A. (1996). Importance measures in global
    sensitivity analysis of nonlinear models. Reliability Engineering &
    System Safety, 52(1), 1–17.
    https://doi.org/10.1016/0951-8320(96)00002-6
    Args:
        p1 (float): Coefficient (optional), with default value 7
        p2 (float): Coefficient (optional), with default value 0.1

    Returns:
        float : Value of first effect (conditional) variance of ishigami function
    """
    V1 = 0.2 * p2 * np.pi ** 4 + 0.02 * p2 ** 2 * np.pi ** 8 + 0.5
    V2 = 0.125 * p1 ** 2
    V3 = 0
    return np.array([V1, V2, V3])


def first_order_indices(p1=P1, p2=P2):
    """First order Sobol' indices of Ishigami test funcion.

    according to (50)-(53) in [1].

    [1] Homma, T., & Saltelli, A. (1996). Importance measures in global
    sensitivity analysis of nonlinear models. Reliability Engineering &
    System Safety, 52(1), 1–17.
    https://doi.org/10.1016/0951-8320(96)00002-6
    Args:
        p1 (float): Coefficient (optional), with default value 7
        p2 (float): Coefficient (optional), with default value 0.1

    Returns:
        float : analytical values of first order Sobol indices of Ishigami function
    """
    V = variance(p1=p1, p2=p2)
    Vi = first_effect_variance(p1=p1, p2=p2)
    return Vi / V


def total_order_indices(p1=P1, p2=P2):
    """Total order Sobol' indices of Ishigami test funcion.

    according to (50)-(57) in [1].

    [1] Homma, T., & Saltelli, A. (1996). Importance measures in global
    sensitivity analysis of nonlinear models. Reliability Engineering &
    System Safety, 52(1), 1–17.
    https://doi.org/10.1016/0951-8320(96)00002-6
    Args:
        p1 (float): Coefficient (optional), with default value 7
        p2 (float): Coefficient (optional), with default value 0.1

    Returns:
        float : analytical values of total order Sobol indices of Ishigami function
    """
    V = variance(p1=p1, p2=p2)
    Vi = first_effect_variance(p1=p1, p2=p2)
    V1 = Vi[0]
    V2 = Vi[1]
    V3 = Vi[2]
    V12 = 0
    V13 = p2 ** 2 * np.pi ** 8 * (1.0 / 18.0 - 0.02)
    V23 = 0
    V123 = 0
    VT1 = V1 + V12 + V13 + V123
    VT2 = V2 + V12 + V23 + V123
    VT3 = V3 + V13 + V23 + V123
    return np.array([VT1, VT2, VT3]) / V


def main(job_id, params):
    """Interface to ishigami function.

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters
    Returns:
        float: Value of ishigami function at parameters specified in input dict
    """
    return ishigami(params['x1'], params['x2'], params['x3'])
