"""Sine test function."""
# pylint: disable=invalid-name

import numpy as np


def sinus_test_fun(x1, **kwargs):
    """A standard sine as a test function.

    Args:
        x1 (float): Input of the sinus in RAD

    Returns:
        float: Value of the sinus function
    """
    result = np.sin(x1, **kwargs)

    return result
