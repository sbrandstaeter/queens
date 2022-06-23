"""Sobol's G function.

A test function for sensitivity analysis.

References:
    [1]: Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S. (2010).
    Variance based sensitivity analysis of model output.
    Design and estimator for the total sensitivity index.
    Computer Physics Communications, 181(2), 259–270.
    https://doi.org/10.1016/j.cpc.2009.09.018
"""
# pylint: disable=invalid-name

import numpy as np

# values  of G*_6 in Table 5 of [1]
dim = 10
A = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.8, 1, 2, 3, 4])
ALPHA = np.array([2] * dim)
DELTA = np.array([0] * dim)


def sobol_g_function(a=A, alpha=ALPHA, delta=DELTA, **kwargs):
    """Compute generalized Sobol's G function.

    with variable dimension. See (33) in [1].
    Default is a 10 dimensional version as defined in [1].

    Args:
        a (ndarray): vector of a parameter values
        alpha (ndarray): vector of alpha parameter values
        delta (ndarray): vector of delta parameter values
        kwargs (dict): contains the input x_i ~ U(0,1) (uniformly from [0,1])

    Returns:
        double: value of Sobol function
    """
    # some safety checks:
    if not np.all(a >= 0):
        raise ValueError("a >= 0 not satisfied")
    if not np.all(alpha > 0):
        raise ValueError("alpha > 0 not satisfied")
    if not (np.all(delta >= 0) and np.all(delta <= 1)):
        raise ValueError("0 <= delta <= 1 not satisfied")

    # From kwargs get the x_i
    x = []
    for key, value in kwargs.items():
        if key.find("x") > -1:
            x.append(value)
    x = np.array(x)

    if not (x.shape == a.shape and x.shape == alpha.shape and x.shape == delta.shape):
        raise ValueError("shape mismatch")

    g = ((1 + alpha) * np.abs(2 * (x + delta - np.floor(x + delta)) - 1) ** alpha + a) / (1 + a)

    return np.prod(g)


def first_effect_variance(a=A, alpha=ALPHA):
    """Calculate first effect variance V_xi[E_x~i[Y|xi]].

    Args:
        a (ndarray): vector of a parameter values
        alpha (ndarray): vector of alpha parameter values

    Returns:
        ndarray: vector of first order variances
    """
    Vi = alpha**2 / (1 + 2 * alpha) / (1 + a) ** 2
    return Vi


def variance(Vi=first_effect_variance(a=A, alpha=ALPHA)):
    """Calculate variance of Sobol function V_x[Y].

    Args:
        Vi (ndarray): vector of first effect variances

    Returns:
        double: variance of Sobol's G function
    """
    V = np.prod(1 + Vi) - 1
    return V


def first_order_indices(a=A, alpha=ALPHA):
    """Compute first order indices of the Sobol test function.

    see (32) in [1].

    Args:
        a (ndarray): vector of a parameter values
        alpha (ndarray): vector of alpha parameter values

    Returns:
        ndarray: vector of first order indices
    """
    Vi = first_effect_variance(a, alpha)
    V = variance(Vi=Vi)
    Si = Vi / V
    return Si


def total_order_indices(a=A, alpha=ALPHA):
    """Compute total indices of Sobol test function.

    see (31)-(32) in [1]

    Args:
        a (ndarray): vector of a parameter values
        alpha (ndarray): vector of alpha parameter values

    Returns:
        ndarray: vector of total order indices
    """
    Vi = first_effect_variance(a=a, alpha=alpha)
    V = variance(Vi=Vi)

    ST = np.empty(Vi.shape)
    for i in range(Vi.shape[0]):
        mask = np.ones(Vi.shape, dtype=bool)
        mask[i] = False
        ST[i] = Vi[i] * np.prod(1 + Vi[mask]) / V

    return ST
