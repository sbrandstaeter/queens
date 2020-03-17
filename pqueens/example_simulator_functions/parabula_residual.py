import numpy as np


def parabula_residual(x1):
    """
    Residual formulation of a parabula

    Args:
        x1 (float):  Input one

    Returns:
        ndarray: Vector of residuals of the parabula
    """

    res1 = 10.0 * x1 - 3.0

    return np.array([res1])


def main(job_id, params):
    """ Interface to Residuals of Rosenbrock banana function

    Args:
        job_id (int):   ID of job
        params (dict):  Dictionary with parameters

    Returns:
        ndarray: Vector of residuals of the parabula at the
                 positions specified in the params dict
    """
    return parabula_residual(params['x1'])
