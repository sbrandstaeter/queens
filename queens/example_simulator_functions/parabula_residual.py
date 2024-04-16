"""Residual of a parabula."""

import numpy as np


def parabula_residual(x1):
    """Residual formulation of a parabula.

    Args:
        x1 (float):  Input parameter 1

    Returns:
        ndarray: Vector of residuals of the parabula
    """
    res1 = 10.0 * x1 - 3.0

    return np.array([res1])
