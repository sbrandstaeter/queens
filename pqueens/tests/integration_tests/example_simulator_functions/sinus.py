"""Sine test function."""
# pylint: disable=invalid-name

import numpy as np


def sinus_test_fun(x1, **kwargs):
    """Evaluate standard sine as a test function.

    Args:
        x1 (float): Input of the sinus in RAD

    Returns:
        result (float): Value of the sinus function
    """
    result = np.sin(x1)
    return result


def gradient_sinus_test_fun(x1, **kwargs):
    """Evaluate sine and its gradient.

    Args:
        x1 (float): Input of the sinus in RAD

    Returns:
        result (float): Value of the sinus function
        gradient (float): Gradient of the sinus function
    """
    result = np.sin(x1)
    gradient = np.cos(x1)
    return result, gradient
