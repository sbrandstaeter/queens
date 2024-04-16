"""Ma test function."""

import numpy as np


def ma09(x1, x2):
    r"""Ma09 function: Two dimensional benchmark for UQ defined in [1].

    :math:`f({\bf x}) = \frac{1}{|0.3-x_1^2 - x_2^2|+0.1}`

    Args:
        x1 (float): Input parameter 1 in [0, 1]
        x2 (float): Input parameter 2 in [0, 1]

    Returns:
        float: Value of the `ma09` function


    References:
        [1] Ma, X., & Zabaras, N. (2009). An adaptive hierarchical sparse grid
            collocation algorithm for the solution of stochastic differential
            equations. Journal of Computational Physics, 228(8), 3084?3113.
    """
    y = 1 / (np.abs(0.3 - x1**2 - x2**2) + 0.1)

    return y
