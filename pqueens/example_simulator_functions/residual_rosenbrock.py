import numpy as np

def residual_rosenbrock(x1, x2):
    """
    Residuals of Rosenbrock banana function

    Args:
        x1 (float):  Input one
        x2 (float):  Input two

    Returns:
        ndarray: Vector of residuals of the Rosenbrock function
    """

    res1 = 10.0 * (x2 - x1 * x1)
    res2 = 1.0 - x1

    return np.array([res1, res2])

def main(job_id, params):
    """ Interface to Residuals of Rosenbrock banana function

    Args:
        job_id (int):   ID of job
        params (dict):  Dictionary with parameters

    Returns:
        ndarray: Vector of residuals of the Rosenbrock function at the
                 positions specified in the params dict
    """
    return residual_rosenbrock(params['x1'], params['x2'])
