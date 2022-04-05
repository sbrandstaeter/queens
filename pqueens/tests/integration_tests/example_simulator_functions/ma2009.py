import numpy as np


def ma2009(x1, x2):
    """Ma09 function: Two dimensional benchmark function for UQ defined in [1]

    :math:`f({\\bf x}) = \\frac{1}{|0.3-x_1^2 - x_2^2|+0.1}`

    Args:
        x1 (float): input parameter 1 in [0, 1]
        x2 (float): input parameter 2 in [0, 1]

    Returns:
        float: value of ma2009 function


    References:

        [1] Ma, X., & Zabaras, N. (2009). An adaptive hierarchical sparse grid
            collocation algorithm for the solution of stochastic differential
            equations. Journal of Computational Physics, 228(8), 3084?3113.
    """

    y = 1 / (np.abs(0.3 - x1**2 - x2**2) + 0.1)

    return y


def main(job_id, params):
    """Interface to ma function.

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters
    Returns:
        float: Value of ma function at parameters specified in input dict
    """
    return ma2009(params['x1'], params['x2'])
