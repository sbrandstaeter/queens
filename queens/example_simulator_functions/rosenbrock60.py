"""Rosenbrock function.

[1]: Rosenbrock, H. H. (1960). An Automatic Method for Finding the
Greatest or Least Value of a      Function. The Computer Journal, 3(3),
175–184. doi:10.1093/comjnl/3.3.175
"""

import numpy as np


def rosenbrock60(x1, x2):
    """Rosenbrocks banana function.

    Args:
        x1 (float):  Input parameter 1
        x2 (float):  Input parameter 2

    Returns:
        float: Value of the Rosenbrock function
    """
    a = 1.0 - x1
    b = x2 - x1 * x1
    return a * a + b * b * 100.0


def rosenbrock60_residual(x1, x2):
    """Residuals of the Rosenbrock banana function.

    Args:
        x1 (float):  Input parameter 1
        x2 (float):  Input parameter 2

    Returns:
        ndarray: Vector of residuals of the Rosenbrock function
    """
    res1 = 10.0 * (x2 - x1 * x1)
    res2 = 1.0 - x1

    return np.array([res1, res2])


def rosenbrock60_residual_1d(x1):
    """Residuals of the Rosenbrock banana function.

    Args:
        x1 (float):  Input parameter 1

    Returns:
        ndarray: Vector of residuals of the Rosenbrock function
    """
    return rosenbrock60_residual(x1, x2=1)
