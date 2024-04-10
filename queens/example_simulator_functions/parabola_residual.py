"""Residual of a parabola."""

import numpy as np


def parabola_residual(x1):
    """Residual formulation of a parabola.

    Args:
        x1 (float):  Input parameter 1

    Returns:
        ndarray: Vector of residuals of the parabola
    """
    res1 = 10.0 * x1 - 3.0

    return np.array([res1])
